"""Per-timestep evaluation of the pitch-class probe on saved diffusion trajectories.

A trajectory file (npz) holds
  z      : (N+1, C, T) latents z_N (pure noise) ... z_0 (clean), or (N, C, T) if z_0 is separate
  step   : (N+1,) sampler step index (0 = first denoising step)
  sigma  : (N+1,) noise level of each latent (whatever the sampler exposes), optional
plus metadata.  This module only needs a callable probe(z) -> probabilities (1, 12, T)
and a reference label (12, T) with a frame mask (T,), e.g. from metrics.pseudo_label.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .reliability import reliability


def _bce(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def frame_metrics(p: np.ndarray, y: np.ndarray, mask: np.ndarray, thr: float = 0.5) -> dict:
    """p, y: (12, T) ; mask: (T,) frames to evaluate.  Micro-F1 over all (class, frame) cells,
    positive-frame recall (frame has an active class AND probe predicts at least one correct
    active class), and BCE."""
    p, y = p[:, mask], y[:, mask]
    pred = p >= thr
    tp = float((pred & (y > 0)).sum())
    fp = float((pred & (y == 0)).sum())
    fn = float((~pred & (y > 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    pos_frames = y.sum(axis=0) > 0
    hit = ((pred & (y > 0)).sum(axis=0) > 0)
    pos_recall = float(hit[pos_frames].mean()) if pos_frames.any() else float("nan")
    top1 = p.argmax(axis=0)
    top1_acc = float((y[top1, np.arange(y.shape[1])] > 0)[pos_frames].mean()) if pos_frames.any() else float("nan")
    return {"f1": f1, "precision": prec, "recall": rec, "pos_recall": pos_recall,
            "top1_acc": top1_acc, "bce": _bce(p, y)}


@torch.no_grad()
def evaluate_trajectory(probe: Callable[[torch.Tensor], torch.Tensor], z: np.ndarray | torch.Tensor,
                        y_ref: np.ndarray, mask: np.ndarray | None = None, device: str = "cpu",
                        thr: float = 0.5, normalize_rms: bool = False) -> list[dict]:
    """Run the probe on every latent of a trajectory.  Returns one dict per row of z, with
    frame metrics vs y_ref, the three reliability scores, and agreement with the probe's own
    prediction on the last (cleanest) latent."""
    zt = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=device)
    if zt.dim() == 2:
        zt = zt.unsqueeze(0)
    T = y_ref.shape[1]
    mask = np.ones(T, dtype=bool) if mask is None else np.asarray(mask, bool)
    probs = []
    for i in range(zt.shape[0]):
        x = zt[i:i + 1]
        if normalize_rms:
            x = x / (x.pow(2).mean().sqrt() + 1e-8)
        p = probe(x)
        if p.shape[-1] != T:                      # align probe frames to the label frames
            p = torch.nn.functional.interpolate(p, size=T, mode="nearest")
        probs.append(p)
    p_final = (probs[-1][0] >= thr).cpu().numpy()
    rows = []
    for i, p in enumerate(probs):
        pn = p[0].cpu().numpy()
        m = frame_metrics(pn, y_ref, mask, thr)
        m.update({
            "row": i,
            "R_entropy": float(reliability(p, "entropy", torch.as_tensor(mask, device=device))[0]),
            "R_margin": float(reliability(p, "margin", torch.as_tensor(mask, device=device))[0]),
            "R_maxprob": float(reliability(p, "maxprob", torch.as_tensor(mask, device=device))[0]),
            "agree_final": float(((pn >= thr) == p_final)[:, mask].mean()),
            "mean_p": float(pn[:, mask].mean()),
        })
        rows.append(m)
    return rows


def summarize(per_traj: list[list[dict]], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a metric across trajectories -> (mean, lo95, hi95) per row."""
    arr = np.array([[r[key] for r in rows] for rows in per_traj], dtype=float)  # (n, rows)
    mean = np.nanmean(arr, axis=0)
    n = np.isfinite(arr).sum(axis=0)
    se = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(n, 1))
    return mean, mean - 1.96 * se, mean + 1.96 * se
