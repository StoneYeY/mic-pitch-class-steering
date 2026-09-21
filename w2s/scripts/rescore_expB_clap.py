"""Fill the CLAP column of an Exp B per_run.csv from the saved wavs and re-aggregate (table.csv).
Used for runs made without CLAP (e.g. the Stable Audio 3 backend env).  No generation.

  EXPB_RES=results/021_sa3_expB EXPB_RUNS=runs/021_sa3_expB SPLIT=test python w2s/scripts/rescore_expB_clap.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from w2s import data, metrics  # noqa: E402
from w2s.scripts.aggregate_expB import aggregate  # noqa: E402

RES = Path(os.environ.get("EXPB_RES", "results/021_sa3_expB"))
RUNS = Path(os.environ.get("EXPB_RUNS", "runs/021_sa3_expB"))
SPLIT = os.environ.get("SPLIT", "test")

df = pd.read_csv(RES / "per_run.csv")
prompts = data.TEST_PROMPTS if SPLIT == "test" else data.DEV_PROMPTS
clap = metrics.ClapScorer()
temb = {pid: clap.text_embed([prompts[int(pid)]])[0] for pid in df.prompt_id.unique()}
vals, missing, cache = [], 0, {}
for _, r in df.iterrows():
    w = RUNS / f"{r.trial}__{r.method}.wav"
    if not w.exists():
        vals.append(np.nan); missing += 1; continue
    y, sr = sf.read(w)
    ae = clap.audio_embed([(y.T if y.ndim > 1 else y, sr)])[0]
    vals.append(float(clap.score(ae[None], temb[r.prompt_id][None])[0]))
df["clap"] = vals
df.to_csv(RES / "per_run.csv", index=False)
print(f"filled clap for {len(df) - missing}/{len(df)} rows ({missing} wavs missing)")
print(df.groupby("method")[["coherence_mlsp", "clap"]].mean().round(3).to_string())
aggregate(RES)
print("DONE")
