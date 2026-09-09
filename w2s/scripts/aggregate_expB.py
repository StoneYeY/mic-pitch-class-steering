"""Aggregate Exp B per_run.csv into the paper table: mean ± 95% CI per method, paired Wilcoxon
vs SAO and vs MLSP-fixed (Holm-corrected), per-melody breakdown.  Runs anywhere (no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from w2s import metrics  # noqa: E402

KEYS = ["coherence_mlsp", "coherence_w2s", "chroma_cos", "clap", "coherence_firstnote", "n_updates", "runtime_s"]


def aggregate(res_dir: Path) -> pd.DataFrame:
    res_dir = Path(res_dir)
    df = pd.read_csv(res_dir / "per_run.csv")
    methods = list(dict.fromkeys(df.method))
    piv = {k: df.pivot(index="trial", columns="method", values=k) for k in KEYS}
    rows = []
    for m in methods:
        rec = dict(method=m, n=int(piv["coherence_mlsp"][m].notna().sum()))
        for k in KEYS:
            mean, lo, hi = metrics.bootstrap_mean_ci(piv[k][m].values)
            rec[k] = mean; rec[f"{k}_lo"] = lo; rec[f"{k}_hi"] = hi
        for ref in ("sao", "mlsp"):
            if ref in methods and ref != m:
                for k in ("coherence_mlsp", "clap"):
                    pr = metrics.paired_wilcoxon(piv[k][m].values, piv[k][ref].values, alternative="two-sided")
                    rec[f"p_{k}_vs_{ref}"] = pr.p_wilcoxon; rec[f"d_{k}_vs_{ref}"] = pr.mean_diff
        rows.append(rec)
    tab = pd.DataFrame(rows)
    for col in [c for c in tab.columns if c.startswith("p_")]:
        tab[col + "_holm"] = metrics.holm(tab[col].values)
    if (res_dir / "fad.csv").exists():
        fad = pd.read_csv(res_dir / "fad.csv").drop(columns=["n"], errors="ignore")
        tab = tab.merge(fad, on="method", how="left")
    tab.to_csv(res_dir / "table.csv", index=False)
    show = ["method", "n", "coherence_mlsp", "chroma_cos", "clap", "n_updates", "runtime_s"]
    show += [c for c in ("fad_clap_maestro", "p_coherence_mlsp_vs_mlsp_holm", "p_clap_vs_sao_holm") if c in tab.columns]
    print(tab[show].round(3).to_string())
    per_mel = df.groupby(["method", "melody"]).coherence_mlsp.mean().unstack().round(3)
    per_mel.to_csv(res_dir / "per_melody.csv"); print("\nper-melody coherence:\n", per_mel.to_string())
    return tab


if __name__ == "__main__":
    aggregate(Path(sys.argv[1] if len(sys.argv) > 1 else "results/expB"))
