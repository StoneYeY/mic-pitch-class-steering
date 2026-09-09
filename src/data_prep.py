"""
Data Preparation for MIC Probe Training

Encodes MAESTRO audio to VAE latents and extracts pitch labels.
Includes diffusion-style noise augmentation.
"""

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import os
import gc

from .utils import (
    find_maestro_pairs,
    midi_to_pitch_class,
    midi_to_piano_roll,
    add_diffusion_noise,
)


class MaestroDataset(torch.utils.data.Dataset):
    """
    Dataset for probe training.

    Each sample contains:
    - latent: VAE encoded audio latent
    - label: Pitch class or piano roll label
    - Optional: noisy versions of latent for augmentation
    """

    def __init__(
        self,
        data,  # str/Path to .pt file, or list of sample dicts
        use_noise_augmentation: bool = True,
        noise_sigmas: List[float] = None,
    ):
        """
        Args:
            data: Path to preprocessed .pt file, or list of sample dicts
            use_noise_augmentation: Whether to use noisy latents
            noise_sigmas: Noise levels for augmentation
        """
        if isinstance(data, (str, Path)):
            # weights_only=False: dataset files are full-pickle dicts, not state_dicts
            self.data = torch.load(data, weights_only=False)
        else:
            self.data = data
        self.use_noise_augmentation = use_noise_augmentation
        self.noise_sigmas = noise_sigmas if noise_sigmas is not None else [0.0, 0.1, 0.3, 0.5, 0.7]
        self.training = False

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        latent = sample['audio_latent']  # (1, C, T)
        label = sample['pitch_label']    # (T, n_classes)

        # Squeeze batch dimension
        if latent.dim() == 3 and latent.shape[0] == 1:
            latent = latent.squeeze(0)  # (C, T)

        # Random noise augmentation during training
        if self.use_noise_augmentation and self.training:
            sigma = np.random.choice(self.noise_sigmas)
            if sigma > 0:
                latent = add_diffusion_noise(latent.unsqueeze(0), sigma).squeeze(0)

        return {
            'latent': latent,
            'label': label,
        }

    def train(self):
        """Set training mode for noise augmentation."""
        self.training = True
        return self

    def eval(self):
        """Set eval mode (no noise augmentation)."""
        self.training = False
        return self


def build_probe_dataset(
    maestro_folder: str,
    pipe,  # StableAudioPipeline
    save_path: str,
    max_files: Optional[int] = None,
    clip_duration: float = 10.0,
    use_pitch_class: bool = True,  # True=12 classes, False=128 (piano roll)
    verbose: bool = True,
) -> List[Dict]:
    """
    Build probe training dataset from MAESTRO.

    1. Load audio segments
    2. Encode to VAE latents
    3. Extract pitch labels from MIDI
    4. Save to .pt file

    Args:
        maestro_folder: Path to MAESTRO dataset
        pipe: Loaded StableAudioPipeline
        save_path: Where to save the dataset
        max_files: Max number of files to process (None = all)
        clip_duration: Duration of each clip in seconds
        use_pitch_class: Use 12 pitch classes (True) or 128 piano roll (False)
        verbose: Print progress

    Returns:
        List of dataset samples
    """
    device = next(iter(pipe.vae.parameters())).device
    dtype = next(iter(pipe.vae.parameters())).dtype

    # Find audio-MIDI pairs
    pairs = find_maestro_pairs(maestro_folder)
    if max_files is not None:
        pairs = pairs[:max_files]

    if verbose:
        print(f"Found {len(pairs)} audio-MIDI pairs")

    dataset = []
    n_classes = 12 if use_pitch_class else 128

    # Get sample rate from VAE
    sample_rate = getattr(pipe.vae.config, 'sampling_rate', 44100)

    for file_idx, (wav_path, midi_path) in enumerate(tqdm(pairs, desc="Processing files")):

        # Load full audio file — librosa handles resampling with polyphase filter (memory-efficient)
        try:
            import librosa
            audio_full, sr = librosa.load(wav_path, sr=sample_rate, mono=False)
            # librosa returns (channels, samples) for mono=False; mono returns 1D
            if audio_full.ndim == 1:
                audio_full = np.stack([audio_full, audio_full], axis=0)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            continue

        # Ensure stereo (2, samples)
        if audio_full.shape[0] == 1:
            audio_full = np.concatenate([audio_full, audio_full], axis=0)
        elif audio_full.shape[0] > 2:
            audio_full = audio_full[:2]

        total_duration = audio_full.shape[1] / sr

        # Process clips
        num_clips = int(total_duration // clip_duration)

        for clip_idx in range(num_clips):
            start_time = clip_idx * clip_duration

            # Extract audio segment
            start_sample = int(start_time * sr)
            end_sample = int((start_time + clip_duration) * sr)
            audio_segment = audio_full[:, start_sample:end_sample]

            # Pad if needed
            target_samples = int(clip_duration * sr)
            if audio_segment.shape[1] < target_samples:
                pad = target_samples - audio_segment.shape[1]
                audio_segment = np.pad(audio_segment, ((0, 0), (0, pad)))

            # Encode to VAE latent
            audio_tensor = torch.from_numpy(audio_segment).unsqueeze(0).to(device, dtype=dtype)

            with torch.no_grad():
                latent = pipe.vae.encode(audio_tensor).latent_dist.sample()
                latent = latent.cpu()

            # KEY FIX: Compute fps from actual latent shape
            latent_len = latent.shape[-1]
            fps = latent_len / clip_duration

            # Extract pitch labels with correct fps
            if use_pitch_class:
                pitch_label = midi_to_pitch_class(midi_path, start_time, clip_duration, fps)
            else:
                pitch_label = midi_to_piano_roll(midi_path, start_time, clip_duration, fps)

            # Ensure label length matches latent length
            if pitch_label.shape[0] != latent_len:
                # Interpolate label to match
                label_tensor = torch.from_numpy(pitch_label).float()
                label_tensor = label_tensor.T.unsqueeze(0)  # (1, n_classes, T)
                label_tensor = nn.functional.interpolate(
                    label_tensor, size=latent_len, mode='nearest'
                )
                pitch_label = label_tensor.squeeze(0).T.numpy()  # (T, n_classes)

            sample = {
                'audio_latent': latent,  # (1, 64, T)
                'pitch_label': torch.from_numpy(pitch_label).float(),  # (T, n_classes)
                'audio_path': wav_path,
                'midi_path': midi_path,
                'clip_index': clip_idx,
                'start_time': start_time,
                'fps': fps,
                'latent_len': latent_len,
            }

            dataset.append(sample)

        # Free audio from memory
        del audio_full

        # Periodic save, GC, and GPU cache clear every 5 files (O9: merged into one block)
        if (file_idx + 1) % 5 == 0:
            torch.save(dataset, save_path)
            gc.collect()
            torch.cuda.empty_cache()
            if verbose:
                print(f"  Saved checkpoint: {len(dataset)} samples")

    # Final save
    torch.save(dataset, save_path)

    if verbose:
        print(f"\nDataset complete: {len(dataset)} samples")
        print(f"Saved to: {save_path}")

        if dataset:
            sample = dataset[0]
            print(f"\nSample info:")
            print(f"  Latent shape: {sample['audio_latent'].shape}")
            print(f"  Label shape: {sample['pitch_label'].shape}")
            print(f"  FPS: {sample['fps']:.2f}")

    return dataset


def load_probe_dataset(
    data_path: str,
    use_noise_augmentation: bool = True,
    noise_sigmas: List[float] = None,
    train_split: float = 0.9,
    n_val_files: Optional[int] = None,
    shuffle_seed: int = 0,
) -> Tuple[MaestroDataset, MaestroDataset]:
    """
    Load and split probe dataset with file-level splitting to prevent data leakage.

    Clips from the same recording share timbre and performer style — splitting at
    the clip level lets the same WAV appear in both train and val, inflating metrics.
    This function groups clips by source file and splits entire files into train/val.

    Args:
        data_path: Path to .pt file
        use_noise_augmentation: Whether to use noise augmentation
        noise_sigmas: Noise levels
        train_split: Fraction of files for training (ignored when n_val_files is set)
        n_val_files: Explicit number of files for validation; use this when the dataset
            is small enough that train_split produces fewer than ~3 val files.
        shuffle_seed: Fixed seed for reproducible file shuffle

    Returns:
        (train_dataset, val_dataset)
    """
    import random
    from collections import defaultdict
    if noise_sigmas is None:
        noise_sigmas = [0.0, 0.1, 0.3, 0.5, 0.7]

    # weights_only=False: dataset files are full-pickle dicts, not state_dicts
    full_data = torch.load(data_path, weights_only=False)

    # Group clips by source WAV file to prevent leakage across the split
    by_file: "defaultdict[str, list]" = defaultdict(list)
    for sample in full_data:
        by_file[sample['audio_path']].append(sample)

    files = list(by_file.keys())
    random.Random(shuffle_seed).shuffle(files)

    if n_val_files is not None:
        n_train_files = len(files) - n_val_files
    else:
        n_train_files = int(len(files) * train_split)

    n_val = len(files) - n_train_files
    n_val_clips = sum(len(by_file[f]) for f in files[n_train_files:])
    if n_val < 3:
        import warnings
        warnings.warn(
            f"Val set has only {n_val} file(s) and {n_val_clips} clips — metrics will have "
            f"high variance. Use n_val_files=3 or collect more data.",
            stacklevel=2,
        )

    train_data = [s for f in files[:n_train_files] for s in by_file[f]]
    val_data = [s for f in files[n_train_files:] for s in by_file[f]]

    train_dataset = MaestroDataset(train_data, use_noise_augmentation, noise_sigmas).train()
    val_dataset = MaestroDataset(val_data, use_noise_augmentation=False).eval()

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test
    print("Data preparation module loaded successfully")
