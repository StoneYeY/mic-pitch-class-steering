"""Exp C: causal steering sensitivity across diffusion time (single-position "burst" intervention).

For each dev trial (5 prompts x 5 melodies x 2 seeds = 50) and each position t in POSITIONS,
run the sampler with guidance ONLY at steps {t, t+1, t+2} (lambda = BURST_LAMBDA), and compare
with the unguided same-seed generation:
  dCoh(t)   = coherence(guided@t) - coherence(unguided)      (MLSP metric and w2s metric)
  dChroma(t), dCLAP(t), logmel_l1(t) (how much the audio moved), dBCE0(t) (probe loss on the
  final latent, a latent-space proxy), plus the R_t observed at t.
Outputs: results/<job>/per_run.csv, by_position.csv (paired mean + bootstrap CI), fig_expC.png.
Wavs stay in runs/<job>/.  Resumable: existing per-run rows are skipped.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from w2s import data, metrics  # noqa: E402
from w2s.evalclip import ClipEvaluator  # noqa: E402
from w2s.generate import W2SGenerator  # noqa: E402
from w2s.mlsp_bridge import load_pipeline, load_probe  # noqa: E402
from w2s.schedules import GuidanceSchedule, make_schedule  # noqa: E402

RES = Path(os.environ.get("W2S_RESULTS", "results/expC")); RES.mkdir(parents=True, exist_ok=True)
RUNS = Path(os.environ.get("W2S_RUNS", "runs/expC")); RUNS.mkdir(parents=True, exist_ok=True)
POSITIONS = [int(x) for x in os.environ.get("EXPC_POSITIONS", "2,6,10,14,18,22,26,30,34,38,42,46").split(",")]
BURST = int(os.environ.get("EXPC_BURST", "3"))
LAM = float(os.environ.get("EXPC_LAMBDA", "0.10"))
SEEDS = [int(x) for x in os.environ.get("EXPC_SEEDS", "0,1").split(",")]
MELODIES = os.environ.get("EXPC_MELODIES", ",".join(data.MELODIES)).split(",")
STEPS = 50


def probe_bce(gen, z_final: np.ndarray, target: np.ndarray, n_aud: int) -> float:
    z = torch.as_tensor(z_final, dtype=torch.float32, device=gen.device)[None]
    tgt = torch.as_tensor(target.T, dtype=torch.float32, device=gen.device)[None]   # (1,T,12)
    with torch.no_grad():
        logits = gen.probe(z)
        return float(F.binary_cross_entropy_with_logits(logits[:, :n_aud], tgt[:, :n_aud]))


def main():
    pipe, probe = load_pipeline(), load_probe()
    gen = W2SGenerator(pipe, probe)
    ev = ClipEvaluator(use_clap=True, device=str(gen.device))
    csv = RES / "per_run.csv"
    done = set()
    if csv.exists():
        old = pd.read_csv(csv)
        done = set(zip(old.trial, old.position))
        rows = old.to_dict("records")
    else:
        rows = []
    trials = data.trial_grid("dev", melodies=MELODIES, seeds=SEEDS)
    print(f"{len(trials)} trials x {len(POSITIONS)} positions, burst={BURST}, lambda={LAM}")
    base_cache: dict[tuple, dict] = {}
    t_start = time.time()
    n_new = 0
    for tr in trials:
        mel = data.MELODIES[tr.melody]
        key = (tr.prompt_id, tr.seed)
        # unguided reference for this (prompt, seed) — shared across melodies
        if key not in base_cache:
            wav = RUNS / f"base-p{tr.prompt_id:02d}-s{tr.seed}.wav"
            r0 = gen.run(tr.prompt, tr.seed, mel, make_schedule("sao"))
            sf.write(str(wav), r0.audio, r0.sr)
            base_cache[key] = dict(audio=r0.audio, final=r0.final_latent, wav=wav, log=r0.log)
        base = base_cache[key]
        tgt_aud = gen.make_target(mel, 5.0)[0][0].T.cpu().numpy()[:, :int(np.ceil(5 * gen.fps))]
        n_aud = tgt_aud.shape[1]
        bkey = (tr.tid, -1)
        if bkey not in done:
            e0 = ev.evaluate(base["audio"], gen.sr, mel, tgt_aud, n_aud, tr.prompt, base["wav"])
            rows.append(dict(trial=tr.tid, prompt_id=tr.prompt_id, melody=tr.melody, seed=tr.seed, position=-1,
                             R_at=np.nan, coherence_mlsp=e0["coherence_mlsp"], coherence_w2s=e0["coherence_w2s"],
                             chroma_cos=e0["chroma_cos"], clap=e0["clap"], logmel_l1=0.0,
                             coherence_firstnote=e0["coherence_firstnote"],
                             bce0=probe_bce(gen, base["final"], np.pad(tgt_aud, ((0, 0), (0, gen.T - n_aud))), n_aud),
                             n_updates=0, runtime_s=np.nan))
            done.add(bkey)
        for pos in POSITIONS:
            if (tr.tid, pos) in done:
                continue
            sched = GuidanceSchedule(f"burst@{pos}", frozenset(range(pos, min(pos + BURST, STEPS))), lam=LAM)
            r = gen.run(tr.prompt, tr.seed, mel, sched)
            wav = RUNS / f"{tr.tid}-t{pos:02d}.wav"
            sf.write(str(wav), r.audio, r.sr)
            e = ev.evaluate(r.audio, gen.sr, mel, tgt_aud, n_aud, tr.prompt, wav, ref_audio=base["audio"])
            rows.append(dict(trial=tr.tid, prompt_id=tr.prompt_id, melody=tr.melody, seed=tr.seed, position=pos,
                             R_at=r.log[pos]["R"], coherence_mlsp=e["coherence_mlsp"], coherence_w2s=e["coherence_w2s"],
                             chroma_cos=e["chroma_cos"], clap=e["clap"], logmel_l1=e["logmel_l1"],
                             coherence_firstnote=e["coherence_firstnote"],
                             bce0=probe_bce(gen, r.final_latent, r.target, n_aud),
                             n_updates=r.n_updates, runtime_s=r.runtime_s))
            done.add((tr.tid, pos)); n_new += 1
            if n_new % 10 == 0:
                pd.DataFrame(rows).to_csv(csv, index=False)
                print(f"  {n_new} new runs, {time.time()-t_start:.0f}s  last: pos={pos} R={r.log[pos]['R']:.2f} "
                      f"coh={e['coherence_mlsp']:.3f} clap={e['clap']:.3f}")
    df = pd.DataFrame(rows); df.to_csv(csv, index=False)
    # paired deltas vs the unguided run of the same trial
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
    bp = pd.DataFrame(out); bp.to_csv(RES / "by_position.csv", index=False)
    print(bp[["position", "R_at", "d_coherence_mlsp", "d_coherence_mlsp_lo", "d_coherence_mlsp_hi", "d_clap", "logmel_l1", "d_bce0"]].round(3).to_string())
    summ = dict(argmax_dcoh_pos=int(bp.position[bp.d_coherence_mlsp.values.argmax()]),
                argmin_dclap_pos=int(bp.position[bp.d_clap.values.argmin()]),
                corr_R_dcoh=float(np.corrcoef(bp.R_at, bp.d_coherence_mlsp)[0, 1]),
                corr_dcoh_dclap=float(np.corrcoef(bp.d_coherence_mlsp, bp.d_clap)[0, 1]),
                corr_dcoh_dbce0=float(np.corrcoef(bp.d_coherence_mlsp, bp.d_bce0)[0, 1]))
    json.dump(summ, open(RES / "summary.json", "w"), indent=2); print(summ)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].errorbar(bp.progress, bp.d_coherence_mlsp, yerr=[bp.d_coherence_mlsp - bp.d_coherence_mlsp_lo, bp.d_coherence_mlsp_hi - bp.d_coherence_mlsp], marker="o")
        ax[0].axhline(0, color="k", lw=.5); ax[0].set_title("Δ coherence (burst@t)")
        ax[1].errorbar(bp.progress, bp.d_clap, yerr=[bp.d_clap - bp.d_clap_lo, bp.d_clap_hi - bp.d_clap], marker="o", color="C1")
        ax[1].axhline(0, color="k", lw=.5); ax[1].set_title("Δ CLAP (burst@t)")
        ax[2].plot(bp.progress, bp.logmel_l1, marker="o", color="C2"); ax[2].set_title("log-mel L1 vs unguided")
        for a in ax: a.set_xlabel("denoising progress"); a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(RES / "fig_expC.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("plot failed:", e)
    print(f"done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
