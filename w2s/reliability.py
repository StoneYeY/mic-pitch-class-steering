"""Target-free reliability of a multi-label pitch-class probe.

The MLSP probe outputs 12 *independent* sigmoid probabilities per latent frame
(multi-label), so a categorical entropy over the raw outputs is ill-defined and,
worse, a probe that confidently predicts "no note anywhere" on pure noise would
look *reliable* under a plain binary-entropy score.  We therefore normalise the
12 outputs of each frame into a distribution q_f = p_f / sum_c p_{f,c} and use
how *peaked* q_f is.  This is insensitive to the overall magnitude of p.

All functions accept torch tensors shaped (B, 12, T) or (12, T) (channel-first,
as produced by a 1x1 conv head) and return per-sample scalars in [0, 1].
"""
from __future__ import annotations

import math
from typing import Literal

import torch

Mode = Literal["entropy", "margin", "maxprob"]

_EPS = 1e-8


def _to_bct(p: torch.Tensor) -> torch.Tensor:
    if p.dim() == 2:
        p = p.unsqueeze(0)
    if p.dim() != 3:
        raise ValueError(f"expected (B,C,T) or (C,T), got {tuple(p.shape)}")
    return p


def frame_normalized_entropy(p: torch.Tensor) -> torch.Tensor:
    """Per-frame entropy of q_f = p_f / sum_c p_{f,c}, normalised by log C.  Returns (B, T)."""
    p = _to_bct(p).float().clamp_min(0.0)
    q = p / (p.sum(dim=1, keepdim=True) + _EPS)
    h = -(q * torch.log(q + _EPS)).sum(dim=1)
    return h / math.log(p.shape[1])


def frame_margin(p: torch.Tensor) -> torch.Tensor:
    """Per-frame top-1 minus top-2 probability.  Returns (B, T)."""
    p = _to_bct(p).float()
    top2 = p.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def reliability(
    p: torch.Tensor,
    mode: Mode = "entropy",
    frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scalar reliability per sample, in [0, 1].

    mode="entropy": R = 1 - mean_f H(q_f)/log C          (recommended, target-free)
    mode="margin":  R = mean_f (top1 - top2)
    mode="maxprob": R = mean_f max_c p_{f,c}
    frame_mask: optional (B, T) or (T,) bool mask of frames to average over
                (e.g. target-active frames only).  Default: all frames.
    """
    p = _to_bct(p)
    if mode == "entropy":
        per_frame = 1.0 - frame_normalized_entropy(p)
    elif mode == "margin":
        per_frame = frame_margin(p)
    elif mode == "maxprob":
        per_frame = p.float().max(dim=1).values
    else:
        raise ValueError(mode)
    if frame_mask is None:
        return per_frame.mean(dim=1)
    m = frame_mask.to(per_frame.dtype)
    if m.dim() == 1:
        m = m.unsqueeze(0).expand_as(per_frame)
    return (per_frame * m).sum(dim=1) / (m.sum(dim=1) + _EPS)


def all_reliabilities(p: torch.Tensor) -> dict[str, float]:
    """Convenience: every mode for a single sample (C,T) or (1,C,T)."""
    p = _to_bct(p)
    assert p.shape[0] == 1, "one sample at a time"
    return {m: float(reliability(p, m)[0]) for m in ("entropy", "margin", "maxprob")}
