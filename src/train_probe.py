"""
Probe Training for MIC

Trains probes to decode pitch information from VAE latents.
This is Contribution 1: proving latent space interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import time
from tqdm import tqdm

from .probe import create_probe, count_parameters
from .data_prep import MaestroDataset
from .utils import add_diffusion_noise


def train_probe(
    probe: nn.Module,
    train_dataset: MaestroDataset,
    val_dataset: Optional[MaestroDataset] = None,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    noise_sigmas: list = None,
    use_noise_augmentation: bool = True,
    save_dir: str = "checkpoints",
    device: str = "cuda",
    verbose: bool = True,
) -> Dict:
    """
    Train probe network.

    Args:
        probe: Probe network (LinearProbe or CNNProbe)
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        noise_sigmas: Noise levels for augmentation
        use_noise_augmentation: Whether to add noise to latents
        save_dir: Directory to save checkpoints
        device: Device to train on
        verbose: Print progress

    Returns:
        Training history dict
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    probe = probe.to(device)
    probe.train()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Loss function: BCE for multi-label classification
    criterion = nn.BCEWithLogitsLoss()

    # Data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    if verbose:
        print("=" * 60)
        print("PROBE TRAINING")
        print("=" * 60)
        print(f"Parameters: {count_parameters(probe):,}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset) if val_dataset else 0}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {num_epochs}")
        print(f"Noise augmentation: {use_noise_augmentation}")
        print("=" * 60)

    if noise_sigmas is None:
        noise_sigmas = [0.0, 0.1, 0.3, 0.5, 0.7]

    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0,
        'best_loss': float('inf'),
    }

    current_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        probe.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            latent = batch['latent'].to(device, dtype=torch.float32)  # (B, C, T)
            label = batch['label'].to(device)    # (B, T, n_classes)

            # Noise augmentation
            if use_noise_augmentation:
                sigma = noise_sigmas[torch.randint(len(noise_sigmas), (1,)).item()]
                if sigma > 0:
                    latent = add_diffusion_noise(latent, sigma)

            # Forward pass
            logits = probe(latent)  # (B, T, n_classes)

            # Ensure shapes match
            if logits.shape[1] != label.shape[1]:
                # Interpolate label to match logits
                label = label.permute(0, 2, 1)  # (B, n_classes, T)
                label = F.interpolate(label, size=logits.shape[1], mode='nearest')
                label = label.permute(0, 2, 1)  # (B, T, n_classes)

            # Loss
            loss = criterion(logits, label)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Scheduler step
        scheduler.step()

        avg_train_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_train_loss)

        # Validation
        val_loss = None
        if val_loader is not None:
            val_loss = evaluate_loss(probe, val_loader, criterion, device)
            history['val_loss'].append(val_loss)

        # Save best model
        current_loss = val_loss if val_loss is not None else avg_train_loss
        if current_loss < history['best_loss']:
            history['best_loss'] = current_loss
            history['best_epoch'] = epoch + 1

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
            }, save_dir / 'probe_best.pt')

        # Logging
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (num_epochs - epoch - 1)

            msg = f"Epoch {epoch+1:3d}/{num_epochs} | Train: {avg_train_loss:.4f}"
            if val_loss is not None:
                msg += f" | Val: {val_loss:.4f}"
            msg += f" | LR: {scheduler.get_last_lr()[0]:.2e}"
            msg += f" | ETA: {eta/60:.1f}min"
            print(msg)

        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': probe.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': current_loss,
            }, save_dir / f'probe_epoch_{epoch+1:03d}.pt')

    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': probe.state_dict(),
        'loss': current_loss,
    }, save_dir / 'probe_final.pt')

    total_time = time.time() - start_time

    if verbose:
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best loss: {history['best_loss']:.4f} (epoch {history['best_epoch']})")
        print(f"Saved to: {save_dir}")
        print("=" * 60)

    return history


def evaluate_loss(
    probe: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """Evaluate probe loss on a dataset."""
    probe.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            latent = batch['latent'].to(device, dtype=torch.float32)
            label = batch['label'].to(device)

            logits = probe(latent)

            # Match shapes
            if logits.shape[1] != label.shape[1]:
                label = label.permute(0, 2, 1)
                label = F.interpolate(label, size=logits.shape[1], mode='nearest')
                label = label.permute(0, 2, 1)

            loss = criterion(logits, label)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def load_probe(checkpoint_path: str, probe_type: str = "cnn", device: str = "cuda", **kwargs) -> nn.Module:
    """Load trained probe from checkpoint."""
    probe = create_probe(probe_type=probe_type, **kwargs)

    import pickle
    try:
        # weights_only=True is preferred; falls back for checkpoints saved with optimizer state.
        # PyTorch <2.5 raises RuntimeError; >=2.5 raises pickle.UnpicklingError.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, AttributeError):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    probe.load_state_dict(checkpoint['model_state_dict'])
    probe = probe.to(device)
    probe.eval()

    loss_val = checkpoint.get('loss', float('nan'))
    print(f"Loaded probe from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {loss_val:.4f}")

    return probe


if __name__ == "__main__":
    print("Train probe module loaded successfully")
