"""Stability of the steering-sensitivity peak (Exp C), no GPU needed.

Reads results/<expC>/per_run.csv and reports: bootstrap distribution of the argmax of the mean
dCoherence(t) curve, per-prompt / per-melody / per-seed / leave-one-prompt-out peaks, the share of
individual trials whose own peak is mid-sampling, and paired mid-vs-early / mid-vs-late tests.

  python w2s/scripts/expC_stability.py results/004_expC_burst [results/015_expC_lam05 ...]
Writes <dir>/stability.json for each directory and prints a summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

STEPS = 50


def analyse(d: Path, n_boot: int = 5000, seed: int = 0) -> dict:
    df = pd.read_csv(d / "per_run.csv")
    base = df[df.position == -1].set_index("trial").coherence_mlsp
    g = df[df.position >= 0].copy()
    g["dcoh"] = g.coherence_mlsp - g.trial.map(base)
    piv = g.pivot_table(index="trial", columns="position", values="dcoh")
    piv.columns = [int(c) for c in piv.columns]
    pos = list(piv.columns)
    meta = g.drop_duplicates("trial").set_index("trial")[["prompt_id", "melody", "seed"]]
    J = piv.join(meta)
    mean_curve = piv.mean()
    rng = np.random.default_rng(seed)
    M = piv.values; n = len(M)
    boots = np.array([pos[int(np.argmax(M[rng.integers(0, n, n)].mean(0)))] for _ in range(n_boot)])
    prog = lambda t: (t + 1) / STEPS  # noqa: E731  paper convention: progress of position t
    early = piv[[p for p in pos if p <= 10]].mean(1)
    mid = piv[[p for p in pos if 22 <= p <= 30]].mean(1)
    late = piv[[p for p in pos if p >= 38]].mean(1)
    tp = piv.idxmax(axis=1)
    out = dict(
        n_trials=int(n), positions=pos, mean_curve={int(k): round(float(v), 4) for k, v in mean_curve.items()},
        peak=int(mean_curve.idxmax()), peak_progress=round(prog(int(mean_curve.idxmax())), 2),
        peak_value=round(float(mean_curve.max()), 4),
        boot_median=float(np.median(boots)), boot_ci=[float(x) for x in np.percentile(boots, [2.5, 97.5])],
        boot_counts={int(k): int(v) for k, v in pd.Series(boots).value_counts().sort_index().items()},
        boot_share_18_34=float(np.mean((boots >= 18) & (boots <= 34))),
        per_prompt={int(k): int(v[pos].mean().idxmax()) for k, v in J.groupby("prompt_id")},
        per_melody={str(k): int(v[pos].mean().idxmax()) for k, v in J.groupby("melody")},
        per_seed={int(k): int(v[pos].mean().idxmax()) for k, v in J.groupby("seed")},
        loo_prompt={int(p): int(J[J.prompt_id != p][pos].mean().idxmax()) for p in sorted(meta.prompt_id.unique())},
        trial_peak_share_18_34=float(((tp >= 18) & (tp <= 34)).mean()),
        mid_minus_early=dict(mean=round(float((mid - early).mean()), 4), p=float(wilcoxon(mid - early).pvalue), wins=float((mid > early).mean())),
        mid_minus_late=dict(mean=round(float((mid - late).mean()), 4), p=float(wilcoxon(mid - late).pvalue), wins=float((mid > late).mean())),
    )
    json.dump(out, open(d / "stability.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        o = analyse(Path(arg))
        print(f"== {arg}: peak {o['peak']} (s={o['peak_progress']}, dC={o['peak_value']}), bootstrap median {o['boot_median']} "
              f"CI {o['boot_ci']}, counts {o['boot_counts']}")
        print("   per prompt", o["per_prompt"], "per melody", o["per_melody"], "LOO", o["loo_prompt"])
        print(f"   mid-early {o['mid_minus_early']}  mid-late {o['mid_minus_late']}  trial peaks in 18-34: {o['trial_peak_share_18_34']:.2f}")
