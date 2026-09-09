"""
Utility functions for MIC project
"""

import torch
import numpy as np
import pretty_midi
from typing import Tuple, List, Optional
from pathlib import Path
import glob


def midi_to_pitch_class(
    midi_path: str,
    start_time: float,
    duration: float,
    fps: float,  # Dynamically computed, NOT hardcoded
    n_classes: int = 12
) -> np.ndarray:
    """
    Extract pitch class labels from MIDI file.

    Args:
        midi_path: Path to MIDI file
        start_time: Start time in seconds
        duration: Duration in seconds
        fps: Frames per second (computed from latent shape)
        n_classes: Number of pitch classes (12 for pitch class, 128 for piano roll)

    Returns:
        pitch_class: (n_frames, n_classes) binary array
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    end_time = start_time + duration
    n_frames = int(duration * fps)

    pitch_labels = np.zeros((n_frames, n_classes), dtype=np.float32)

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            # Skip notes outside time range
            if note.end <= start_time or note.start >= end_time:
                continue

            # Calculate frame indices
            note_start_rel = max(0, note.start - start_time)
            note_end_rel = min(duration, note.end - start_time)

            start_frame = int(note_start_rel * fps)
            end_frame = int(note_end_rel * fps)

            # FIX: Ensure at least 1 frame for short notes
            end_frame = max(end_frame, start_frame + 1)
            end_frame = min(end_frame, n_frames)

            # Get pitch class (or direct pitch for piano roll)
            if n_classes == 12:
                pitch_idx = note.pitch % 12
            else:
                pitch_idx = note.pitch
                if pitch_idx >= n_classes:
                    continue

            pitch_labels[start_frame:end_frame, pitch_idx] = 1.0

    return pitch_labels


def midi_to_piano_roll(
    midi_path: str,
    start_time: float,
    duration: float,
    fps: float
) -> np.ndarray:
    """Extract full piano roll (128 keys) from MIDI."""
    return midi_to_pitch_class(midi_path, start_time, duration, fps, n_classes=128)


def notes_to_pitch_class_target(
    notes: List[int],
    tempo: float,
    note_duration_beats: float,
    total_frames: int,
    audio_length: float,
    device: torch.device
) -> torch.Tensor:
    """
    Convert MIDI note list to target pitch class tensor for inference.

    Args:
        notes: List of MIDI pitches [60, 62, 64, ...]
        tempo: BPM
        note_duration_beats: Duration of each note in beats
        total_frames: Number of frames (matches latent T dimension)
        audio_length: Total audio duration in seconds
        device: Target device

    Returns:
        target: (1, T, 12) tensor
    """
    # FIX: Use audio_length as total duration, not melody duration
    fps = total_frames / audio_length

    sec_per_beat = 60.0 / tempo
    note_sec = note_duration_beats * sec_per_beat

    target = torch.zeros(1, total_frames, 12, device=device)

    t = 0.0
    for note in notes:
        if t >= audio_length:
            break

        start_frame = int(t * fps)
        end_frame = int((t + note_sec) * fps)

        # FIX: Ensure at least 1 frame
        end_frame = max(end_frame, start_frame + 1)
        end_frame = min(end_frame, total_frames)

        pitch_class = note % 12
        target[0, start_frame:end_frame, pitch_class] = 1.0

        t += note_sec

    return target


def add_diffusion_noise(
    latent: torch.Tensor,
    sigma: float,
    normalize_output: bool = True
) -> torch.Tensor:
    """
    Add diffusion-style noise to latent.

    Uses: z_noisy = z + sigma * epsilon
    Then optionally RMS normalizes to preserve scale.

    Args:
        latent: Clean latent (B, C, T)
        sigma: Noise level
        normalize_output: Whether to RMS normalize

    Returns:
        Noisy latent
    """
    if sigma == 0.0:
        return latent.clone()

    noise = torch.randn_like(latent)
    noisy = latent + sigma * noise

    if normalize_output:
        # RMS normalize to preserve scale
        orig_rms = latent.pow(2).mean(dim=(-1, -2), keepdim=True).sqrt()
        noisy_rms = noisy.pow(2).mean(dim=(-1, -2), keepdim=True).sqrt()
        noisy = noisy / (noisy_rms + 1e-8) * orig_rms

    return noisy


def find_maestro_pairs(maestro_folder: str) -> List[Tuple[str, str]]:
    """Find all matching WAV + MIDI file pairs in MAESTRO dataset."""
    wav_files = glob.glob(f'{maestro_folder}/**/*.wav', recursive=True)

    pairs = []
    for wav_path in wav_files:
        p = Path(wav_path)
        midi_path = p.with_suffix('.midi')
        if not midi_path.exists():
            midi_path = p.with_suffix('.mid')
        if midi_path.exists():
            pairs.append((str(p), str(midi_path)))

    return sorted(pairs)


def compute_rms(x: torch.Tensor, dim=None, keepdim=True) -> torch.Tensor:
    """Compute RMS of tensor."""
    if dim is None:
        return x.pow(2).mean().sqrt()
    return x.pow(2).mean(dim=dim, keepdim=keepdim).sqrt()


def rms_normalize(x: torch.Tensor, target_rms: float = 1.0) -> torch.Tensor:
    """Normalize tensor to target RMS."""
    current_rms = compute_rms(x)
    return x / (current_rms + 1e-8) * target_rms
