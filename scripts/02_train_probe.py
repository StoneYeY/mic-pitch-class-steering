#!/usr/bin/env python
"""
Step 2: Train pitch class probe on VAE latents.

This script trains a probe to decode pitch information
from Stable Audio VAE latents, demonstrating that the
latent space encodes interpretable musical concepts.

Contribution 1: Interpretable Latent Space

Usage:
    python scripts/02_train_probe.py --data_path data/probe_dataset.pt --probe_type cnn
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch

import config
from src.probe import create_probe, count_parameters
from src.data_prep import load_probe_dataset
from src.train_probe import train_probe
from src.evaluate_probe import evaluate_probe, evaluate_noise_robustness


def main():
    parser = argparse.ArgumentParser(description="Train pitch class probe")
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(config.DATA_DIR / "probe_dataset.pt"),
        help="Path to probe dataset"
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="cnn",
        choices=["linear", "cnn"],
        help="Probe architecture type"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--no_noise_augmentation",
        action="store_true",
        help="Disable noise augmentation during training"
    )
    parser.add_argument(
        "--out_classes",
        type=int,
        default=12,
        help="Number of output classes (12 for pitch class, 128 for piano roll)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(config.CHECKPOINT_DIR),
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--n_val_files",
        type=int,
        default=None,
        help="Explicit number of source files held out for validation (overrides train_split)"
    )
    parser.add_argument(
        "--probe_metrics_path",
        type=str,
        default=str(config.OUTPUT_DIR / "evaluation" / "probe_metrics.json"),
        help="Path to save Contribution 1 metrics JSON"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MIC - PROBE TRAINING")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Probe type: {args.probe_type}")
    print(f"Output classes: {args.out_classes}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Noise augmentation: {not args.no_noise_augmentation}")
    print("=" * 60)

    # Setup
    config.setup_memory_efficient()
    device = config.get_device()
    print(f"\nDevice: {device}")

    # Load dataset
    print("\nLoading dataset...")
    train_dataset, val_dataset = load_probe_dataset(
        data_path=args.data_path,
        use_noise_augmentation=not args.no_noise_augmentation,
        noise_sigmas=config.NOISE_SIGMAS,
        train_split=0.9,
        n_val_files=args.n_val_files,
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create probe
    print(f"\nCreating {args.probe_type} probe...")
    probe = create_probe(
        probe_type=args.probe_type,
        in_channels=config.PROBE_IN_CHANNELS,
        hidden_channels=config.PROBE_HIDDEN_CHANNELS,
        out_classes=args.out_classes,
    )
    print(f"Parameters: {count_parameters(probe):,}")

    # Train
    history = train_probe(
        probe=probe,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        noise_sigmas=config.NOISE_SIGMAS,
        use_noise_augmentation=not args.no_noise_augmentation,
        save_dir=args.save_dir,
        device=str(device),
        verbose=True,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION ON VALIDATION SET")
    print("=" * 60)

    # Load best model
    best_path = Path(args.save_dir) / "probe_best.pt"
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    probe.load_state_dict(checkpoint['model_state_dict'])
    probe = probe.to(device)
    probe.eval()

    # Evaluate
    results = evaluate_probe(
        probe=probe,
        dataset=val_dataset,
        device=str(device),
        verbose=True
    )

    # Test noise robustness
    print("\n" + "=" * 60)
    print("NOISE ROBUSTNESS TEST")
    print("=" * 60)
    print("(This determines safe guidance timestep range)")

    noise_results = evaluate_noise_robustness(
        probe=probe,
        dataset=val_dataset,
        noise_sigmas=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
        device=str(device),
        verbose=True
    )

    # Summary
    print("\n" + "=" * 60)
    print("CONTRIBUTION 1 SUMMARY")
    print("=" * 60)
    print(f"Probe type: {args.probe_type}")
    print(f"Best validation loss: {history['best_loss']:.4f}")
    print(f"F1 (micro): {results['f1_micro']:.4f}")
    print(f"Positive recall: {results['positive_recall']:.4f}")
    print(f"Frame overlap: {results['frame_overlap']:.4f}")
    print("\nConclusion:")
    if results['f1_micro'] > 0.3:
        print("  VAE latent space encodes interpretable pitch information.")
    else:
        print("  Probe performance is low. Consider more training or different architecture.")
    print("=" * 60)

    # Persist Contribution 1 metrics (E3)
    probe_metrics = {
        'probe_type': args.probe_type,
        'dataset': args.data_path,
        'n_train_clips': len(train_dataset),
        'n_val_clips': len(val_dataset),
        'n_val_files': args.n_val_files,
        'best_val_loss': history['best_loss'],
        'best_epoch': history['best_epoch'],
        'f1_micro': results['f1_micro'],
        'f1_macro': results['f1_macro'],
        'precision_micro': results['precision_micro'],
        'recall_micro': results['recall_micro'],
        'positive_recall': results['positive_recall'],
        'positive_f1': results['positive_f1'],
        'frame_overlap': results['frame_overlap'],
        'per_class_f1': results['per_class_f1'],
        'noise_robustness': {
            str(sigma): {'f1_micro': v['f1_micro'], 'positive_recall': v['positive_recall']}
            for sigma, v in noise_results.items()
        },
    }

    metrics_path = Path(args.probe_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(probe_metrics, f, indent=2)
    print(f"\nProbe metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
