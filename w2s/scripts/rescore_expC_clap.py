"""Job 010: fill in the CLAP column of Exp C from the saved wavs (CLAP was broken during the
original run) and recompute by_position.csv with d_clap.  No generation — just re-eval."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from w2s import metrics  # noqa: E402

RESC = Path(os.environ.get("EXPC_RES", "results/004_expC_burst"))
RUNSC = Path(os.environ.get("EXPC_RUNS", "runs/004_expC_burst"))
STEPS = 50

df = pd.read_csv(RESC / "per_run.csv")
clap = metrics.ClapScorer()
print(f"rescoring {len(df)} rows from {RUNSC}")

def wav_for(r):
    if int(r.position) < 0:
        return RUNSC / f"base-p{int(r.prompt_id):02d}-s{int(r.seed)}.wav"
    return RUNSC / f"{r.trial}-t{int(r.position):02d}.wav"

emb_cache, missing = {}, 0
prompts = {}
from w2s import data
for pid in df.prompt_id.unique():
    prompts[pid] = data.DEV_PROMPTS[int(pid)]
tempb = {p: clap.text_embed([prompts[p]])[0] for p in prompts}

claps = []
for _, r in df.iterrows():
    w = wav_for(r)
    if not w.exists():
        claps.append(np.nan); missing += 1; continue
    y, sr = sf.read(w)
    ae = clap.audio_embed([(y.T if y.ndim > 1 else y, sr)])[0]
    claps.append(float(clap.score(ae[None], tempb[r.prompt_id][None])[0]))
df["clap"] = claps
df.to_csv(RESC / "per_run.csv", index=False)
print(f"filled clap for {len(df)-missing}/{len(df)} rows ({missing} wavs missing)")

# recompute by_position with d_clap
base = df[df.position == -1].set_index("trial")
g = df[df.position >= 0].copy()
for k in ("coherence_mlsp", "coherence_w2s", "chroma_cos", "clap", "bce0", "coherence_firstnote"):
    g[f"d_{k}"] = g[k].values - base.loc[g.trial, k].values
out = []
for pos, gg in g.groupby("position"):
    rec = dict(position=pos, progress=(pos + 1) / STEPS, n=len(gg), R_at=gg.R_at.mean())
    for k in ("d_coherence_mlsp", "d_coherence_w2s", "d_chroma_cos", "d_clap", "d_bce0", "logmel_l1", "d_coherence_firstnote"):
        m, lo, hi = metrics.bootstrap_mean_ci(gg[k].values)
        rec.update({k: m, f"{k}_lo": lo, f"{k}_hi": hi})
    out.append(rec)
bp = pd.DataFrame(out); bp.to_csv(RESC / "by_position.csv", index=False)
print(bp[["position", "progress", "R_at", "d_coherence_mlsp", "d_clap", "logmel_l1"]].round(3).to_string())
print("argmax d_coh at progress", float(bp.progress[bp.d_coherence_mlsp.values.argmax()]),
      "| argmin d_clap at progress", float(bp.progress[bp.d_clap.values.argmin()]))
print("DONE")
