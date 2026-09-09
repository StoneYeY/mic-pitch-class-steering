"""Guidance schedules with a matched update budget K.

A schedule answers two questions at every sampler step t (0-based, 0 = pure noise):
  * should_guide(t, R_t, n_used) -> bool      "when"
  * strength(t, R_t)             -> lambda_t  "how strongly"

Fixed windows (Early / Mid / Late / Uniform / MLSP) have a constant lambda.
RAPG-cal fixes *when* offline (Top-K of a dev-set profile w(t)) and adapts *how
strongly* online (lambda_t = lambda * R_t / r_bar).
RAPG-on decides both online: guide iff R_t >= eta and budget not exhausted.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import numpy as np

FIXED_KINDS = ("early", "mid", "late", "uniform", "mlsp")


# --------------------------------------------------------------------------- step sets
def fixed_steps(kind: str, N: int = 50, K: int = 15) -> list[int]:
    """Step sets used in the paper (N=50, K=15): early 0-14, mid 18-32, late 35-49,
    uniform round(linspace(0,N-1,K)), mlsp range(int(0.4N), N, 2)."""
    kind = kind.lower()
    if kind == "early":
        return list(range(0, K))
    if kind == "mid":
        start = N // 2 - K // 2
        return list(range(start, start + K))
    if kind == "late":
        return list(range(N - K, N))
    if kind == "uniform":
        return sorted({int(round(x)) for x in np.linspace(0, N - 1, K)})
    if kind == "mlsp":
        return list(range(int(0.4 * N), N, 2))
    raise ValueError(f"unknown fixed schedule '{kind}'")


def topk_steps(profile: Sequence[float], K: int, tie: str = "earliest") -> list[int]:
    """Top-K steps of a per-step profile w(t) (length N).  NaN -> never selected.
    Ties are broken toward the earliest step (tie='earliest') or latest ('latest')."""
    w = np.asarray(profile, dtype=float)
    w = np.where(np.isnan(w), -np.inf, w)
    idx = np.arange(len(w))
    order = np.lexsort((idx if tie == "earliest" else -idx, -w))  # primary: -w asc == w desc
    chosen = sorted(int(i) for i in order[:K] if np.isfinite(w[i]))
    return chosen


def interpolate_profile(positions: Sequence[int], values: Sequence[float], N: int = 50,
                        smooth: int = 0) -> np.ndarray:
    """Linear interpolation of a profile measured at a few steps onto all N steps
    (constant extrapolation outside), optionally box-smoothed with window `smooth`."""
    pos = np.asarray(positions, dtype=float)
    val = np.asarray(values, dtype=float)
    o = np.argsort(pos)
    w = np.interp(np.arange(N), pos[o], val[o])
    if smooth and smooth > 1:
        k = np.ones(smooth) / smooth
        w = np.convolve(np.pad(w, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"), k, mode="valid")
    return w


# --------------------------------------------------------------------------- schedule object
@dataclass
class GuidanceSchedule:
    name: str
    steps: frozenset[int] | None = None   # fixed step set; None -> online rule
    lam: float = 0.05                     # base strength (MLSP default)
    adaptive_strength: bool = False       # lambda_t = lam * R_t / r_bar
    r_bar: float = 1.0                    # dev-set mean R over guided steps (normalisation)
    eta: float | None = None              # online threshold on R_t
    budget: int | None = None             # cap on #updates (online); None -> len(steps)
    N: int = 50
    meta: dict = field(default_factory=dict)

    # ---- "when"
    def should_guide(self, step: int, R_t: float | None, n_used: int) -> bool:
        if self.steps is not None:
            return step in self.steps
        if self.eta is None:
            return False
        if self.budget is not None and n_used >= self.budget:
            return False
        return R_t is not None and float(R_t) >= self.eta

    # ---- "how strongly"
    def strength(self, step: int, R_t: float | None) -> float:
        if not self.adaptive_strength or R_t is None:
            return self.lam
        return self.lam * float(R_t) / max(self.r_bar, 1e-6)

    # ---- bookkeeping
    @property
    def K(self) -> int | None:
        return len(self.steps) if self.steps is not None else self.budget

    @property
    def is_null(self) -> bool:
        return self.steps is not None and len(self.steps) == 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = sorted(self.steps) if self.steps is not None else None
        return d


def make_schedule(kind: str, N: int = 50, K: int = 15, lam: float = 0.05, *,
                  profile: Sequence[float] | None = None, eta: float | None = None,
                  r_bar: float = 1.0, adaptive_strength: bool | None = None,
                  tie: str = "earliest", name: str | None = None) -> GuidanceSchedule:
    """Factory.

    kind: 'sao' | 'early' | 'mid' | 'late' | 'uniform' | 'mlsp'
          | 'rapg_cal'  (needs profile; Top-K of w(t); adaptive strength on by default)
          | 'rapg_on'   (needs eta; online threshold + budget K; adaptive strength on by default)
          | 'fixed_adaptive' (Top-K of profile with CONSTANT lambda: isolates the 'when' effect)
    """
    kind = kind.lower()
    if kind == "sao":
        return GuidanceSchedule(name or "SAO", frozenset(), lam=0.0, N=N)
    if kind in FIXED_KINDS:
        steps = fixed_steps(kind, N, K)
        if kind == "mlsp" and len(steps) != K:
            print(f"[schedules] warning: MLSP schedule has {len(steps)} updates, K={K}")
        return GuidanceSchedule(name or kind.capitalize() if kind != "mlsp" else (name or "MLSP-fixed"),
                                frozenset(steps), lam=lam,
                                adaptive_strength=bool(adaptive_strength), r_bar=r_bar, N=N)
    if kind in ("rapg_cal", "fixed_adaptive"):
        if profile is None:
            raise ValueError(f"{kind} needs a per-step profile w(t)")
        steps = topk_steps(profile, K, tie=tie)
        ad = (kind == "rapg_cal") if adaptive_strength is None else adaptive_strength
        return GuidanceSchedule(name or ("RAPG-cal" if kind == "rapg_cal" else "TopK-fixed"),
                                frozenset(steps), lam=lam, adaptive_strength=ad, r_bar=r_bar, N=N,
                                meta={"profile": [float(x) for x in profile], "tie": tie})
    if kind == "rapg_on":
        if eta is None:
            raise ValueError("rapg_on needs eta (use calibrate_eta on dev-set R_t)")
        ad = True if adaptive_strength is None else adaptive_strength
        return GuidanceSchedule(name or "RAPG-on", None, lam=lam, adaptive_strength=ad,
                                r_bar=r_bar, eta=float(eta), budget=K, N=N)
    raise ValueError(f"unknown schedule kind '{kind}'")


# --------------------------------------------------------------------------- calibration
def realized_counts(R_dev: np.ndarray, eta: float, cap: int | None) -> np.ndarray:
    """#updates per dev sample under the online rule (threshold eta, optional cap)."""
    R = np.asarray(R_dev, dtype=float)               # (n, N)
    hits = (R >= eta).astype(int)
    if cap is None:
        return hits.sum(axis=1)
    cum = np.cumsum(hits, axis=1)
    return np.minimum(cum[:, -1], cap)


def calibrate_eta(R_dev: np.ndarray, K: int, cap: bool = True, tol: float = 0.05) -> float:
    """Threshold eta such that the mean realized #updates on the dev set is ~K.
    Bisection over eta in [min R, max R]."""
    R = np.asarray(R_dev, dtype=float)
    lo, hi = float(np.nanmin(R)), float(np.nanmax(R))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        c = realized_counts(R, mid, K if cap else None).mean()
        if abs(c - K) <= tol:
            return mid
        if c > K:
            lo = mid      # too many updates -> raise threshold
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_reliability_over(R_dev: np.ndarray, steps: Iterable[int] | None = None,
                          eta: float | None = None) -> float:
    """r_bar: mean R over the guided steps (fixed set) or over threshold hits (online)."""
    R = np.asarray(R_dev, dtype=float)
    if steps is not None:
        s = sorted(steps)
        return float(np.nanmean(R[:, s])) if s else 1.0
    if eta is not None:
        m = R >= eta
        return float(R[m].mean()) if m.any() else 1.0
    return float(np.nanmean(R))


def describe(schedules: Iterable[GuidanceSchedule], N: int = 50) -> str:
    """ASCII strip of the guided steps, for logs."""
    lines = []
    for s in schedules:
        if s.steps is None:
            row = "online: R_t >= %.3f, cap %s" % (s.eta if s.eta is not None else float("nan"), s.budget)
        else:
            row = "".join("#" if t in s.steps else "." for t in range(N)) + f"  K={len(s.steps)}"
        lines.append(f"{s.name:>12s} {row}")
    return "\n".join(lines)
