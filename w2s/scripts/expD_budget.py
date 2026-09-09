"""Exp D: budget sweep. coherence-CLAP Pareto for Uniform, Late and RAPG-cal(dC) at K in {5,10,15,25}.
Subset: 10 test prompts x 5 melodies x 1 seed = 50 trials per (method, K). Resumable."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from w2s import data, schedules as S  # noqa: E402
from w2s.evalclip import ClipEvaluator  # noqa: E402
from w2s.generate import W2SGenerator  # noqa: E402
from w2s.mlsp_bridge import load_pipeline, load_probe  # noqa: E402

RES = Path(os.environ.get("W2S_RESULTS", "results/expD")); RES.mkdir(parents=True, exist_ok=True)
RUNS = Path(os.environ.get("W2S_RUNS", "runs/expD")); RUNS.mkdir(parents=True, exist_ok=True)
EXPC = Path(os.environ.get("EXPC_DIR", "results/004_expC_burst"))
KS = [int(x) for x in os.environ.get("EXPD_KS", "5,10,15,25").split(",")]
PROMPTS = list(range(int(os.environ.get("EXPD_NPROMPTS", "10"))))
LAM = float(os.environ.get("LAMBDA", "0.05"))
STEPS = 50


def main():
    prof_dc = None
    if (EXPC / "by_position.csv").exists():
        bp = pd.read_csv(EXPC / "by_position.csv").sort_values("position")
        prof_dc = S.interpolate_profile(bp.position.values, bp.d_coherence_mlsp.values, STEPS, smooth=3)
    pipe, probe = load_pipeline(), load_probe()
    gen = W2SGenerator(pipe, probe)
    ev = ClipEvaluator(use_clap=True, device=str(gen.device))
    trials = data.trial_grid("test", melodies=list(data.MELODIES), seeds=[0], prompt_ids=PROMPTS)
    csv = RES / "per_run.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["method"], r["K"], r["trial"]) for r in rows}
    print(f"{len(trials)} trials x {len(KS)} K x 2 methods (uniform vs TopK-dC, constant lambda)")
    t0 = time.time(); n = 0
    for K in KS:
        methods = {f"uniform_K{K}": S.make_schedule("uniform", STEPS, K, LAM)}
        if prof_dc is not None:   # concentrated at the steering sweet spot, CONSTANT lambda (no r_bar dependence)
            methods[f"topk_dc_K{K}"] = S.make_schedule("fixed_adaptive", STEPS, K, LAM, profile=prof_dc, name=f"TopK-dC-K{K}")
        for tr in trials:
            mel = data.MELODIES[tr.melody]
            n_aud = int(np.ceil(5 * gen.fps))
            tgt_aud = gen.make_target(mel, 5.0)[0][0].T.cpu().numpy()[:, :n_aud]
            for mname, sch in methods.items():
                if (mname, K, tr.tid) in done:
                    continue
                r = gen.run(tr.prompt, tr.seed, mel, sch)
                wav = RUNS / f"{tr.tid}__{mname}.wav"; sf.write(str(wav), r.audio, r.sr)
                e = ev.evaluate(r.audio, gen.sr, mel, tgt_aud, n_aud, tr.prompt, wav)
                rows.append(dict(method=mname.rsplit("_K", 1)[0], K=K, trial=tr.tid, melody=tr.melody,
                                 coherence_mlsp=e["coherence_mlsp"], chroma_cos=e["chroma_cos"], clap=e["clap"],
                                 n_updates=r.n_updates))
                n += 1
                if n % 25 == 0:
                    pd.DataFrame(rows).to_csv(csv, index=False)
                    print(f"  {n} runs {time.time()-t0:.0f}s last {mname} coh={e['coherence_mlsp']:.3f} clap={e['clap']:.3f}")
    df = pd.DataFrame(rows); df.to_csv(csv, index=False)
    g = df.groupby(["method", "K"]).agg(coh=("coherence_mlsp", "mean"), coh_sd=("coherence_mlsp", "std"),
                                        clap=("clap", "mean"), chroma=("chroma_cos", "mean"), n=("trial", "count")).reset_index()
    g.to_csv(RES / "pareto.csv", index=False)
    print(g.round(3).to_string())
    json.dump(g.to_dict("records"), open(RES / "pareto.json", "w"), indent=1)
    print(f"done {n} new runs in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
