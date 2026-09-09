"""
Probe Evaluation for MIC

Evaluates probes to demonstrate Contribution 1:
VAE latent space encodes interpretable pitch information.

Metrics:
- F1 (micro/macro)
- Positive recall (crucial for sparse labels)
- Frame overlap accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from .probe import create_probe
from .data_prep import MaestroDataset
from .utils import add_diffusion_noise


PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def evaluate_probe(
    probe: nn.Module,
    dataset: MaestroDataset,
    threshold: float = 0.5,
    device: str = "cuda",
    batch_size: int = 32,
    verbose: bool = True,
) -> Dict:
    """
    Comprehensive probe evaluation.

    This is Contribution 1: demonstrating that VAE latents
    encode interpretable pitch information.

    Args:
        probe: Trained probe network
        dataset: Evaluation dataset
        threshold: Classification threshold
        device: Device
        batch_size: Batch size for evaluation
        verbose: Print results

    Returns:
        Dictionary with all metrics
    """
    probe = probe.to(device)
    probe.eval()
    dataset.eval()

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", disable=not verbose):
            latent = batch['latent'].to(device, dtype=torch.float32)
            label = batch['label'].to(device)

            logits = probe(latent)
            probs = torch.sigmoid(logits)

            # Match shapes
            if logits.shape[1] != label.shape[1]:
                label = label.permute(0, 2, 1)
                label = F.interpolate(label, size=logits.shape[1], mode='nearest')
                label = label.permute(0, 2, 1)

            preds = (probs > threshold).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)    # (N, T, C)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, T, C)
    all_probs = np.concatenate(all_probs, axis=0)    # (N, T, C)

    # Flatten for sklearn metrics
    preds_flat = all_preds.reshape(-1, all_preds.shape[-1])   # (N*T, C)
    labels_flat = all_labels.reshape(-1, all_labels.shape[-1])
    probs_flat = all_probs.reshape(-1, all_probs.shape[-1])

    # Compute metrics
    results = compute_all_metrics(preds_flat, labels_flat, probs_flat, verbose)

    return results


def compute_all_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Compute comprehensive evaluation metrics.

    Args:
        preds: Binary predictions (N, C)
        labels: Ground truth labels (N, C)
        probs: Prediction probabilities (N, C)
        verbose: Print results

    Returns:
        Dictionary with all metrics
    """
    n_classes = labels.shape[1]

    # 1. F1 scores
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)

    # 2. Precision and Recall
    precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
    recall_micro = recall_score(labels, preds, average='micro', zero_division=0)

    # 3. Positive recall (crucial for sparse labels)
    # Only consider frames where ground truth has at least one active pitch
    positive_mask = labels.sum(axis=1) > 0
    if positive_mask.sum() > 0:
        positive_recall = recall_score(
            labels[positive_mask],
            preds[positive_mask],
            average='micro',
            zero_division=0
        )
        positive_f1 = f1_score(
            labels[positive_mask],
            preds[positive_mask],
            average='micro',
            zero_division=0
        )
    else:
        positive_recall = 0.0
        positive_f1 = 0.0

    # 4. Frame overlap accuracy
    # For each frame, check if any predicted pitch matches any true pitch
    frame_overlap = compute_frame_overlap(preds, labels)

    # 5. Per-class metrics
    per_class_f1 = []
    per_class_recall = []
    per_class_precision = []

    for i in range(n_classes):
        class_labels = labels[:, i]
        class_preds = preds[:, i]

        # Only compute if class has positive samples
        if class_labels.sum() > 0:
            per_class_f1.append(f1_score(class_labels, class_preds, zero_division=0))
            per_class_recall.append(recall_score(class_labels, class_preds, zero_division=0))
            per_class_precision.append(precision_score(class_labels, class_preds, zero_division=0))
        else:
            per_class_f1.append(np.nan)
            per_class_recall.append(np.nan)
            per_class_precision.append(np.nan)

    results = {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'positive_recall': positive_recall,
        'positive_f1': positive_f1,
        'frame_overlap': frame_overlap,
        'per_class_f1': per_class_f1,
        'per_class_recall': per_class_recall,
        'per_class_precision': per_class_precision,
    }

    if verbose:
        print_evaluation_results(results, n_classes)

    return results


def compute_frame_overlap(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute frame-level overlap accuracy.

    A frame is correct if ANY predicted pitch class matches
    ANY ground truth pitch class.

    This is more lenient than exact match and useful for
    measuring if the probe captures the general pitch content.
    """
    # For each frame, check overlap
    has_overlap = ((preds * labels).sum(axis=1) > 0).astype(float)

    # Only consider frames with ground truth
    has_label = labels.sum(axis=1) > 0

    if has_label.sum() > 0:
        overlap_accuracy = has_overlap[has_label].mean()
    else:
        overlap_accuracy = 0.0

    return overlap_accuracy


def print_evaluation_results(results: Dict, n_classes: int = 12):
    """Print formatted evaluation results."""
    print("\n" + "=" * 60)
    print("PROBE EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- Overall Metrics ---")
    print(f"  F1 (micro):        {results['f1_micro']:.4f}")
    print(f"  F1 (macro):        {results['f1_macro']:.4f}")
    print(f"  Precision (micro): {results['precision_micro']:.4f}")
    print(f"  Recall (micro):    {results['recall_micro']:.4f}")

    print("\n--- Positive-Only Metrics ---")
    print(f"  Positive Recall:   {results['positive_recall']:.4f}")
    print(f"  Positive F1:       {results['positive_f1']:.4f}")
    print(f"  Frame Overlap:     {results['frame_overlap']:.4f}")

    print("\n--- Per-Class F1 ---")
    if n_classes == 12:
        for i, name in enumerate(PITCH_NAMES):
            f1 = results['per_class_f1'][i]
            recall = results['per_class_recall'][i]
            if not np.isnan(f1):
                print(f"  {name:3s}: F1={f1:.3f}, Recall={recall:.3f}")
            else:
                print(f"  {name:3s}: (no samples)")

    # Summary
    valid_f1 = [f for f in results['per_class_f1'] if not np.isnan(f)]
    if valid_f1:
        print(f"\n  Mean per-class F1: {np.mean(valid_f1):.4f}")

    print("=" * 60)


def evaluate_noise_robustness(
    probe: nn.Module,
    dataset: MaestroDataset,
    noise_sigmas: List[float] = None,
    device: str = "cuda",
    verbose: bool = True
) -> Dict:
    """
    Evaluate probe performance at different noise levels.

    This helps determine which denoising steps are safe for
    inference-time guidance.

    Args:
        probe: Trained probe
        dataset: Evaluation dataset
        noise_sigmas: Noise levels to test
        device: Device
        verbose: Print results

    Returns:
        Dict mapping sigma -> metrics
    """
    probe = probe.to(device)
    probe.eval()

    if noise_sigmas is None:
        noise_sigmas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

    results = {}

    for sigma in noise_sigmas:
        if verbose:
            print(f"\nEvaluating at noise sigma = {sigma:.2f}")

        # Create noisy version of dataset
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for sample in tqdm(dataset, disable=not verbose):
                latent = sample['latent'].unsqueeze(0).to(device, dtype=torch.float32)
                label = sample['label']

                # Add noise
                if sigma > 0:
                    latent = add_diffusion_noise(latent, sigma)

                logits = probe(latent)

                # Align label length to logits (B4: same fix as evaluate_probe)
                lbl = label.unsqueeze(0).to(device)  # (1, T_label, C)
                if logits.shape[1] != lbl.shape[1]:
                    lbl = lbl.permute(0, 2, 1)
                    lbl = F.interpolate(lbl, size=logits.shape[1], mode='nearest')
                    lbl = lbl.permute(0, 2, 1)
                label = lbl.squeeze(0).cpu()

                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                preds = (probs > 0.5).astype(float)

                all_preds.append(preds)
                all_labels.append(label.numpy())

        preds_flat = np.concatenate(all_preds, axis=0)
        labels_flat = np.concatenate(all_labels, axis=0)

        # Compute key metrics
        f1 = f1_score(labels_flat, preds_flat, average='micro', zero_division=0)
        positive_mask = labels_flat.sum(axis=1) > 0
        if positive_mask.sum() > 0:
            pos_recall = recall_score(
                labels_flat[positive_mask],
                preds_flat[positive_mask],
                average='micro',
                zero_division=0
            )
        else:
            pos_recall = 0.0

        results[sigma] = {
            'f1_micro': f1,
            'positive_recall': pos_recall,
        }

        if verbose:
            print(f"  F1: {f1:.4f}, Positive Recall: {pos_recall:.4f}")

    if verbose:
        print("\n--- Noise Robustness Summary ---")
        print("Sigma | F1     | Pos Recall")
        print("-" * 30)
        for sigma in noise_sigmas:
            m = results[sigma]
            print(f"{sigma:5.2f} | {m['f1_micro']:.4f} | {m['positive_recall']:.4f}")

    return results


if __name__ == "__main__":
    print("Evaluate probe module loaded successfully")
