#!/usr/bin/env python
"""
Step 1: Prepare probe training dataset from MAESTRO.

This script:
1. Loads MAESTRO audio files
2. Encodes them to Stable Audio VAE latents
3. Extracts pitch class labels from MIDI
4. Saves dataset for probe training

Usage:
    python scripts/01_prepare_data.py --maestro_path /path/to/maestro --max_files 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from diffusers import StableAudioPipeline
from diffusers.models import AutoencoderOobleck

import config
from src.data_prep import build_probe_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare probe training dataset")
    parser.add_argument(
        "--maestro_path",
        type=str,
        required=True,
        help="Path to MAESTRO dataset folder"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(config.DATA_DIR / "probe_dataset.pt"),
        help="Output path for dataset"
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Max number of files to process (None = all)"
    )
    parser.add_argument(
        "--clip_duration",
        type=float,
        default=10.0,
        help="Duration of each clip in seconds"
    )
    parser.add_argument(
        "--use_piano_roll",
        action="store_true",
        help="Use 128-key piano roll instead of 12 pitch classes"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MIC - DATA PREPARATION")
    print("=" * 60)
    print(f"MAESTRO path: {args.maestro_path}")
    print(f"Output path: {args.output_path}")
    print(f"Max files: {args.max_files or 'all'}")
    print(f"Clip duration: {args.clip_duration}s")
    print(f"Label type: {'piano roll (128)' if args.use_piano_roll else 'pitch class (12)'}")
    print("=" * 60)

    # Setup memory optimization
    config.setup_memory_efficient()
    device = config.get_device()
    print(f"\nDevice: {device}")

    # Load ONLY the VAE (saves ~6GB RAM vs full pipeline)
    print("\nLoading Stable Audio VAE only (memory efficient)...")
    vae = AutoencoderOobleck.from_pretrained(
        config.MODEL_ID,
        subfolder="vae",
        torch_dtype=config.DTYPE,
        local_files_only=True,
    )
    vae = vae.to(device)
    print("VAE loaded!")

    # Wrap VAE in a mock pipe object (data_prep only uses pipe.vae)
    class PipeMock:
        pass
    pipe = PipeMock()
    pipe.vae = vae

    # Check GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # Build dataset
    print("\nBuilding dataset...")
    dataset = build_probe_dataset(
        maestro_folder=args.maestro_path,
        pipe=pipe,
        save_path=args.output_path,
        max_files=args.max_files,
        clip_duration=args.clip_duration,
        use_pitch_class=not args.use_piano_roll,
        verbose=True,
    )

    print(f"\nDataset saved to: {args.output_path}")
    print(f"Total samples: {len(dataset)}")

    # Print sample info
    if dataset:
        sample = dataset[0]
        print(f"\nSample info:")
        print(f"  Latent shape: {sample['audio_latent'].shape}")
        print(f"  Label shape: {sample['pitch_label'].shape}")
        print(f"  FPS: {sample['fps']:.2f}")


if __name__ == "__main__":
    main()
