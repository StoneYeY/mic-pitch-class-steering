#!/usr/bin/env python
"""
Step 0 (optional): Generate on-policy probe training data.

Motivation
----------
The MAESTRO-trained probe (script 01) has a domain gap: it was trained on
VAE latents from real piano recordings, but at inference time it sees latents
from Stable Audio's own denoising trajectory — a different distribution.

On-policy training closes this gap:
  1. Generate audio with Stable Audio (no guidance, varied seeds & prompts)
  2. Transcribe each clip with PYIN to get frame-level pitch-class labels
  3. Re-encode the generated audio through the VAE → same latent distribution
     as inference
  4. Save in the same format as the MAESTRO dataset; merge for retraining

The resulting probe trains on latents it will actually encounter at test time.

Usage
-----
    # Generate 500 clips (overnight run):
    python scripts/00_generate_onpolicy.py --n_seeds 56

    # Quick smoke-test (3 clips):
    python scripts/00_generate_onpolicy.py --n_seeds 1 --n_inference_steps 20

    # After generation, retrain probe on combined data:
    python scripts/02_train_probe.py \\
        --dataset_path data/probe_dataset_onpolicy.pt \\
        --n_val_files 4 --num_epochs 100
"""

import argparse
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import soundfile as sf
import librosa
from diffusers import StableAudioPipeline
from diffusers.models import AutoencoderOobleck
from tqdm import tqdm

import config


# Same 9 piano prompts used in inference evaluation.
PIANO_PROMPTS = [
    "Fast-paced Western Classical music with lively violin harmony, electric cello, piano and string accompaniments that are colourful, well-layered, dense, rich, full, pleasant, cheerful, merry and elegant.",
    "An alternative/indie song with groovy bass, punchy drums, piano chords, and synth melodies that create an addictive and retro sound.",
    "Classical music piece with a gentle piano tune and a theremin playing the main melody, creating a unique and heart-touching atmosphere suitable for an animation movie/TV series.",
    "Upbeat indie rock instrumental with piano melody, simple percussion, distortion guitar, and bass.",
    "A slow and captivating piano piece that evokes sombre and reflective emotions with simple yet effective motifs.",
    "Dramatic contemporary classical piano piece with accentuated playing style suitable for documentary or mystery/horror video game soundtrack.",
    "A piano arpeggio melody with reverb and environmental sounds, suitable for amateur video intros or outros.",
    "Relaxing instrumental with piano and French horn, featuring an ascending progression and arpeggiated chords.",
    "A melodic and emotional duet with piano, electric guitar, drums, and synthesizer arrangements that could be suitable for children.",
]


def transcribe_pitch_class(audio: np.ndarray, sr: int, fps: float,
                            n_frames: int) -> np.ndarray:
    """
    PYIN pitch tracking → per-frame 12-class pitch label.

    Returns (n_frames, 12) float32 array. Frames with no detected pitch
    get an all-zero row (silence / unvoiced).
    """
    mono = audio[0] if audio.ndim == 2 else audio
    hop_length = int(sr / fps)

    f0, voiced_flag, _ = librosa.pyin(
        mono,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length,
    )

    midi_float = librosa.hz_to_midi(f0)  # NaN where unvoiced
    label = np.zeros((n_frames, 12), dtype=np.float32)

    for t in range(min(len(midi_float), n_frames)):
        if np.isfinite(midi_float[t]):
            pc = int(round(midi_float[t])) % 12
            label[t, pc] = 1.0

    return label


def encode_audio(audio: np.ndarray, vae, device, dtype) -> torch.Tensor:
    """
    Encode a (2, samples) stereo numpy array through the Stable Audio VAE.
    Returns latent tensor of shape (1, 64, T).
    """
    tensor = torch.from_numpy(audio).unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.sample()
    return latent.cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Generate on-policy probe training data via Stable Audio + PYIN"
    )
    parser.add_argument(
        "--output_path", type=str,
        default=str(config.DATA_DIR / "probe_dataset_onpolicy.pt"),
        help="Where to save the generated dataset",
    )
    parser.add_argument(
        "--merge_with", type=str,
        default=str(config.DATA_DIR / "probe_dataset_5s.pt"),
        help="Existing dataset to merge with (pass '' to skip merging)",
    )
    parser.add_argument(
        "--merged_output_path", type=str,
        default=str(config.DATA_DIR / "probe_dataset_combined.pt"),
        help="Output path for merged (MAESTRO + on-policy) dataset",
    )
    parser.add_argument(
        "--n_seeds", type=int, default=10,
        help="Seeds per prompt. Total clips = n_seeds × n_prompts × clips_per_gen. "
             "clips_per_gen = audio_length // 5  (e.g. 10s → 2 clips)",
    )
    parser.add_argument(
        "--seed_start", type=int, default=1000,
        help="Base seed (seeds = seed_start, seed_start+1, ...)",
    )
    parser.add_argument(
        "--audio_length", type=float, default=10.0,
        help="Generation length in seconds. Longer = more 5s sub-clips per call.",
    )
    parser.add_argument(
        "--clip_duration", type=float, default=5.0,
        help="Sub-clip duration in seconds (must match probe training).",
    )
    parser.add_argument(
        "--n_inference_steps", type=int, default=50,
        help="Denoising steps (fewer = faster but lower quality).",
    )
    parser.add_argument(
        "--save_audio", action="store_true",
        help="Also write generated WAVs to outputs/onpolicy_audio/",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MIC - ON-POLICY DATA GENERATION")
    print("=" * 60)
    n_prompts = len(PIANO_PROMPTS)
    clips_per_gen = int(args.audio_length // args.clip_duration)
    total_clips = args.n_seeds * n_prompts * clips_per_gen
    print(f"Prompts:          {n_prompts}")
    print(f"Seeds per prompt: {args.n_seeds}  (base seed={args.seed_start})")
    print(f"Audio length:     {args.audio_length}s → {clips_per_gen} × {args.clip_duration}s clips each")
    print(f"Target clips:     {total_clips}")
    print(f"Inference steps:  {args.n_inference_steps}")
    print("=" * 60)

    config.setup_memory_efficient()
    device = config.get_device()
    print(f"\nDevice: {device}")

    audio_out_dir = None
    if args.save_audio:
        audio_out_dir = config.OUTPUT_DIR / "onpolicy_audio"
        audio_out_dir.mkdir(parents=True, exist_ok=True)

    # Load full pipeline (need it for generation; reuse VAE from it)
    print("\nLoading Stable Audio pipeline...")
    pipe = StableAudioPipeline.from_pretrained(
        config.MODEL_ID, torch_dtype=config.DTYPE
    )
    pipe = pipe.to(device)
    print("Pipeline loaded.")

    vae = pipe.vae
    sr = getattr(vae.config, 'sampling_rate', config.SAMPLE_RATE)

    dataset = []
    skipped_silent = 0

    seeds = [args.seed_start + i for i in range(args.n_seeds)]
    gen_iter = [(p_idx, prompt, seed)
                for seed in seeds
                for p_idx, prompt in enumerate(PIANO_PROMPTS)]

    for p_idx, prompt, seed in tqdm(gen_iter, desc="Generating"):

        # Generate audio (no probe guidance — pure on-policy baseline)
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=args.n_inference_steps,
                guidance_scale=4.0,
                audio_end_in_s=args.audio_length,
                generator=torch.Generator(device).manual_seed(seed),
            )
        # output.audios: (1, channels, samples) float32 on CPU
        audio_np = output.audios[0].float().cpu().numpy()  # (channels, samples)
        del output
        torch.cuda.empty_cache()

        if audio_out_dir is not None:
            sf.write(
                str(audio_out_dir / f"p{p_idx+1}_s{seed}.wav"),
                audio_np.T, sr
            )

        # Slice into sub-clips and process each
        clip_samples = int(args.clip_duration * sr)
        for clip_idx in range(clips_per_gen):
            start = clip_idx * clip_samples
            end = start + clip_samples
            if end > audio_np.shape[1]:
                break

            clip_audio = audio_np[:, start:end]  # (2, clip_samples)

            # Encode through VAE → latent
            latent = encode_audio(clip_audio, vae, device, config.DTYPE)
            # latent: (1, 64, T)
            n_frames = latent.shape[-1]
            fps = n_frames / args.clip_duration

            # Transcribe with PYIN → pitch class label
            label = transcribe_pitch_class(clip_audio, sr, fps, n_frames)
            label_tensor = torch.from_numpy(label)  # (T, 12)

            # Skip near-silent clips (all-zero label = no pitched content)
            if label_tensor.sum() == 0:
                skipped_silent += 1
                continue

            dataset.append({
                'audio_latent':  latent,                          # (1, 64, T)
                'pitch_label':   label_tensor,                    # (T, 12)
                'source':        'onpolicy',
                'prompt_idx':    p_idx,
                'seed':          seed,
                'clip_index':    clip_idx,
                'fps':           fps,
                'latent_len':    n_frames,
            })

        # Periodic checkpoint
        if len(dataset) > 0 and len(dataset) % 100 == 0:
            torch.save(dataset, args.output_path)
            gc.collect()
            torch.cuda.empty_cache()
            tqdm.write(f"  checkpoint: {len(dataset)} clips saved "
                       f"(skipped {skipped_silent} silent)")

    torch.save(dataset, args.output_path)
    print(f"\nOn-policy dataset: {len(dataset)} clips  "
          f"(skipped {skipped_silent} silent)")
    print(f"Saved to: {args.output_path}")

    # Merge with MAESTRO dataset
    merge_path = args.merge_with.strip() if args.merge_with else ""
    if merge_path and Path(merge_path).exists() and dataset:
        print(f"\nMerging with MAESTRO dataset: {merge_path}")
        maestro_data = torch.load(merge_path, weights_only=False)
        # Tag source if missing
        for s in maestro_data:
            s.setdefault('source', 'maestro')
        combined = maestro_data + dataset
        torch.save(combined, args.merged_output_path)
        print(f"Combined dataset: {len(combined)} clips "
              f"({len(maestro_data)} MAESTRO + {len(dataset)} on-policy)")
        print(f"Saved to: {args.merged_output_path}")
    elif merge_path and not Path(merge_path).exists():
        print(f"\nNote: merge source not found at {merge_path} — skipping merge.")

    print("\nNext step: retrain probe on combined dataset:")
    print(f"  python scripts/02_train_probe.py \\")
    print(f"      --dataset_path {args.merged_output_path or args.output_path} \\")
    print(f"      --n_val_files 4 --num_epochs 100")


if __name__ == "__main__":
    main()
