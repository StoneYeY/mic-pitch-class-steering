"""Control and quality metrics + paired statistics.

Control
  melodic_coherence  : pYIN pitch class vs target in target-active voiced frames (MLSP definition)
  chroma_cosine      : mean cosine between chroma_cqt frames and the binary target chroma
Pseudo-labels
  pseudo_label       : (12,T) pitch-class labels + voiced mask from the *generated* audio
                       (used to score the probe on real diffusion trajectories)
Quality
  ClapScorer         : LAION-CLAP (music) text-audio cosine; also yields audio embeddings
  frechet_distance   : FAD between two embedding sets (use CLAP audio embeddings -> "FAD_clap")
Stats
  paired_wilcoxon, bootstrap_mean_ci, holm

Audio is float mono at its native rate; SAO output is 44.1 kHz stereo -> mono first.
Latent frame rate for SAO's VAE: 44100 / 2048 ≈ 21.53 Hz (hop 2048 samples).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

LATENT_HOP = 2048            # samples per latent frame at 44.1 kHz
LATENT_FPS = 44100 / LATENT_HOP


# --------------------------------------------------------------------------- helpers
def to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=0 if y.shape[0] <= 2 else 1)
    return y


def _align_T(x: np.ndarray, T: int) -> np.ndarray:
    """Resample the last axis of x to length T by nearest-index picking."""
    src = x.shape[-1]
    if src == T:
        return x
    idx = np.minimum((np.arange(T) * src / T).astype(int), src - 1)
    return x[..., idx]


# --------------------------------------------------------------------------- pYIN pitch classes
def pyin_pitch_classes(y: np.ndarray, sr: int, hop: int = LATENT_HOP, fmin: float = 65.4,
                       fmax: float = 2093.0, frame_length: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Frame-level (pc, voiced) from pYIN at hop `hop` samples (default = latent frame rate).
    pc is int in [0,12) (-1 where unvoiced); voiced is bool."""
    import librosa
    y = to_mono(y)
    f0, vflag, vprob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length,
                                   hop_length=hop, fill_na=np.nan)
    voiced = np.asarray(vflag, dtype=bool) & np.isfinite(f0)
    pc = np.full(len(f0), -1, dtype=int)
    pc[voiced] = (np.round(librosa.hz_to_midi(f0[voiced])).astype(int)) % 12
    return pc, voiced


def melodic_coherence(pc: np.ndarray, voiced: np.ndarray, target: np.ndarray,
                      min_voiced_frac: float = 0.2) -> float:
    """Fraction of voiced frames inside target-active regions whose detected pitch class is
    in the target.  NaN if fewer than `min_voiced_frac` of target-active frames are voiced
    (the MLSP exclusion rule; no trial was excluded in practice)."""
    T = target.shape[1]
    pc, voiced = _align_T(pc, T), _align_T(voiced, T)
    active = target.sum(axis=0) > 0
    if active.sum() == 0:
        return float("nan")
    v = active & voiced
    if v.sum() < min_voiced_frac * active.sum():
        return float("nan")
    hit = target[pc[v], np.where(v)[0]] > 0
    return float(hit.mean())


def pseudo_label(pc: np.ndarray, voiced: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    """(12,T) one-hot pitch-class labels from pYIN of the generated audio + voiced mask (T,)."""
    pc, voiced = _align_T(pc, T), _align_T(voiced, T)
    y = np.zeros((12, T), dtype=np.float32)
    idx = np.where(voiced)[0]
    y[pc[idx], idx] = 1.0
    return y, voiced.astype(bool)


# --------------------------------------------------------------------------- chroma cosine
def chroma_cosine(y: np.ndarray, sr: int, target: np.ndarray, hop: int = LATENT_HOP) -> float:
    """Mean per-frame cosine similarity between chroma_cqt and the binary target, over
    target-active frames."""
    import librosa
    y = to_mono(y)
    C = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)         # (12, T')
    T = target.shape[1]
    C = _align_T(C, T)
    active = target.sum(axis=0) > 0
    num = (C * target).sum(axis=0)
    den = np.linalg.norm(C, axis=0) * np.linalg.norm(target, axis=0) + 1e-8
    return float((num / den)[active].mean()) if active.any() else float("nan")


# --------------------------------------------------------------------------- CLAP
class ClapScorer:
    """LAION-CLAP via transformers.  Default checkpoint: laion/larger_clap_music (48 kHz)."""

    def __init__(self, model_id: str = "laion/larger_clap_music", device: str | None = None):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = ClapProcessor.from_pretrained(model_id)
        self.sr = 48000

    def _resample(self, y: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        y = to_mono(y)
        return librosa.resample(y, orig_sr=sr, target_sr=self.sr) if sr != self.sr else y

    @staticmethod
    def _emb_tensor(o):
        """Extract the 512-d projected embedding from get_*_features output across transformers
        versions: a plain tensor (<=4.x), or a BaseModelOutputWithPooling with .pooler_output (5.x)."""
        import torch
        if torch.is_tensor(o):
            return o
        for attr in ("audio_embeds", "text_embeds", "pooler_output"):
            v = getattr(o, attr, None)
            if v is not None:
                return v
        if isinstance(o, (tuple, list)):
            return o[0]
        raise TypeError(f"cannot extract embedding from {type(o)}")

    def audio_embed(self, audios: Sequence[tuple[np.ndarray, int]], batch: int = 8) -> np.ndarray:
        import torch
        out = []
        for i in range(0, len(audios), batch):
            chunk = [self._resample(y, sr) for y, sr in audios[i:i + batch]]
            inp = self.proc(audio=chunk, sampling_rate=self.sr, return_tensors="pt", padding=True)
            keep = {k: v.to(self.device) for k, v in inp.items() if k in ("input_features", "is_longer", "attention_mask")}
            with torch.no_grad():
                e = self._emb_tensor(self.model.get_audio_features(**keep))
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out, 0)

    def text_embed(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        inp = self.proc(text=list(texts), return_tensors="pt", padding=True)
        keep = {k: v.to(self.device) for k, v in inp.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            e = self._emb_tensor(self.model.get_text_features(**keep))
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    def score(self, audio_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
        """Row-wise cosine similarity (embeddings already normalised)."""
        return (audio_emb * text_emb).sum(axis=-1)


# --------------------------------------------------------------------------- FAD
def frechet_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Fréchet distance between Gaussians fitted to two embedding sets (n, D)."""
    from scipy import linalg
    a, b = np.asarray(emb_a, np.float64), np.asarray(emb_b, np.float64)
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(ca @ cb, disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(ca.shape[0]) * 1e-6
        covmean = linalg.sqrtm((ca + eps) @ (cb + eps))
    covmean = covmean.real
    d = mu_a - mu_b
    return float(d @ d + np.trace(ca) + np.trace(cb) - 2.0 * np.trace(covmean))


# --------------------------------------------------------------------------- statistics
@dataclass
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_lo: float
    ci_hi: float
    p_wilcoxon: float

    def __str__(self) -> str:
        return (f"n={self.n} A={self.mean_a:.3f} B={self.mean_b:.3f} diff={self.mean_diff:+.3f} "
                f"[{self.ci_lo:+.3f},{self.ci_hi:+.3f}] p={self.p_wilcoxon:.2g}")


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_wilcoxon(a: np.ndarray, b: np.ndarray, alternative: str = "greater") -> PairedResult:
    """Paired test that A > B (alternative='greater'), on samples with both values finite."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = a - b
    mean, lo, hi = bootstrap_mean_ci(d)
    if len(d) < 5 or np.allclose(d, 0):
        p = float("nan")
    else:
        p = float(wilcoxon(a, b, alternative=alternative, zero_method="wilcox").pvalue)
    return PairedResult(int(len(d)), float(a.mean()), float(b.mean()), mean, lo, hi, p)


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs pass through)."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    adj = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj.tolist()
