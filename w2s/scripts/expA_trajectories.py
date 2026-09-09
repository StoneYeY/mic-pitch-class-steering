"""Exp A: probe reliability along REAL SAO sampling trajectories.

50 unguided generations (5 dev prompts x 10 seeds; unguided runs do not depend on the melody),
trajectories saved (fp16).  For every step t the probe is scored against pseudo-labels from
the generated audio (pYIN at the latent frame rate, audible frames only):
  f1, pos_recall, top1_acc, bce, R_entropy, R_margin, R_maxprob, agree_final
on (a) the raw latent z_t (what MLSP feeds the probe), (b) RMS-normalised z_t, and
(c) the MLSP Gaussian proxy: z_0 + sigma_rel*rms(z_0)*eps with the SAME relative noise level as
step t, RMS-renormalised (this is what MLSP Table 1 measured).
Outputs (results/<job>/): per_step.csv (run x step x variant), summary.csv (mean/CI per step),
corr.json, fig_expA.png.  Trajectories + wavs stay in runs/<job>/.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from w2s import data, metrics, probe_eval  # noqa: E402
from w2s.generate import W2SGenerator  # noqa: E402
from w2s.mlsp_bridge import load_pipeline, load_probe  # noqa: E402
from w2s.schedules import make_schedule  # noqa: E402

RES = Path(os.environ.get("W2S_RESULTS", "results/expA")); RES.mkdir(parents=True, exist_ok=True)
RUNS = Path(os.environ.get("W2S_RUNS", "runs/expA")); RUNS.mkdir(parents=True, exist_ok=True)
N_SEEDS = int(os.environ.get("EXPA_SEEDS", "10"))
STEPS = 50


def main():
    pipe, probe = load_pipeline(), load_probe()
    gen = W2SGenerator(pipe, probe)
    sao = make_schedule("sao")
    n_aud = int(np.ceil(5.0 * gen.fps))
    rows, meta = [], []
    t_start = time.time()
    for pid, prompt in enumerate(data.DEV_PROMPTS):
        for seed in range(N_SEEDS):
            tag = f"dev-p{pid:02d}-s{seed}"
            npz = RUNS / f"{tag}.npz"
            if npz.exists():
                d = np.load(npz)
                traj, audio = d["z"], d["audio"]
            else:
                r = gen.run(prompt, seed, None, sao, save_trajectory=True)
                traj, audio = r.trajectory, r.audio
                sf.write(str(RUNS / f"{tag}.wav"), audio, r.sr)
                np.savez_compressed(npz, z=traj, audio=audio.astype(np.float32),
                                    sigma=np.array([x["sigma"] for x in r.log]), rms=np.array([x["rms_z"] for x in r.log]))
                print(f"{tag}: generated in {r.runtime_s:.1f}s (vram {r.peak_vram_gb:.1f} GB)")
            d = np.load(npz)
            sigma, rms = d["sigma"], d["rms"]
            # pseudo-labels from the generated audio
            pc, voiced = metrics.pyin_pitch_classes(metrics.to_mono(audio.T), gen.sr)
            y_ref, mask = metrics.pseudo_label(pc, voiced, n_aud)
            if mask.sum() < 0.2 * n_aud:
                print(f"{tag}: only {int(mask.sum())}/{n_aud} voiced frames -> skipped"); continue
            y_full = np.zeros((12, gen.T), np.float32); y_full[:, :n_aud] = y_ref
            m_full = np.zeros(gen.T, bool); m_full[:n_aud] = mask
            z0 = traj[-1].astype(np.float32)
            rms0 = float(np.sqrt((z0 ** 2).mean()))
            rng = np.random.default_rng(seed)
            # (c) Gaussian proxy at matched relative noise level per step
            sig_rel = np.sqrt(np.maximum(rms[:, None] ** 2 - rms0 ** 2, 0.0)).ravel() / rms0   # noise share of rms(z_t)
            proxy = np.stack([z0 + s * rms0 * rng.standard_normal(z0.shape).astype(np.float32) for s in sig_rel])
            proxy = proxy / np.sqrt((proxy ** 2).mean(axis=(1, 2), keepdims=True)) * rms0
            variants = {"raw": (traj, False), "rmsnorm": (traj, True), "gauss_proxy": (proxy, False)}
            for vname, (z, norm) in variants.items():
                ev = probe_eval.evaluate_trajectory(gen.probe_fn, z, y_full, m_full, device=str(gen.device), normalize_rms=norm)
                for e in ev:
                    e.update(run=tag, prompt_id=pid, seed=seed, variant=vname, sigma=float(sigma[e["row"]]),
                             rms=float(rms[e["row"]]), sig_rel=float(sig_rel[e["row"]]), progress=(e["row"] + 1) / STEPS)
                    rows.append(e)
            meta.append(dict(run=tag, voiced=int(mask.sum()), n_aud=n_aud, rms0=rms0))
            print(f"{tag}: voiced {int(mask.sum())}/{n_aud}  f1(raw) first/mid/last = "
                  f"{rows[-3*STEPS]['f1']:.2f}/{rows[-3*STEPS+STEPS//2]['f1']:.2f}/{rows[-2*STEPS-1]['f1']:.2f}  "
                  f"[{time.time()-t_start:.0f}s]")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "per_step.csv", index=False)
    keys = ["f1", "pos_recall", "top1_acc", "bce", "R_entropy", "R_margin", "R_maxprob", "agree_final", "mean_p"]
    g = df.groupby(["variant", "row"])
    summ = g[keys].mean().join(g[keys].sem().add_suffix("_sem")).join(g[["progress", "sigma", "rms", "sig_rel"]].mean())
    summ.reset_index().to_csv(RES / "summary.csv", index=False)
    corr = {}
    for v in variants:
        s = summ.loc[v]
        corr[v] = {f"corr({r},f1)": float(np.corrcoef(s[r], s["f1"])[0, 1]) for r in ("R_entropy", "R_margin", "R_maxprob")}
        corr[v]["argmax_f1_step"] = int(s["f1"].values.argmax())
        corr[v]["f1_first_mid_last"] = [float(s["f1"].iloc[0]), float(s["f1"].iloc[STEPS // 2]), float(s["f1"].iloc[-1])]
    # per-run correlation between R_entropy(t) and f1(t) (raw)
    per_run = [float(np.corrcoef(x["R_entropy"], x["f1"])[0, 1]) for _, x in df[df.variant == "raw"].groupby("run")]
    corr["per_run_corr_Rentropy_f1"] = dict(mean=float(np.nanmean(per_run)), median=float(np.nanmedian(per_run)))
    json.dump(dict(corr=corr, runs=meta), open(RES / "corr.json", "w"), indent=2)
    print(json.dumps(corr, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for v, ls in (("raw", "-"), ("rmsnorm", "--"), ("gauss_proxy", ":")):
            s = summ.loc[v]
            axes[0].plot(s["progress"], s["f1"], ls, label=v)
            axes[1].plot(s["progress"], s["R_entropy"], ls, label=v)
            axes[2].plot(s["progress"], s["bce"], ls, label=v)
        for ax, t in zip(axes, ("probe micro-F1 vs pseudo-labels", "reliability R_t (entropy)", "BCE")):
            ax.set_title(t); ax.set_xlabel("denoising progress"); ax.grid(alpha=.3)
        axes[0].legend(); fig.tight_layout(); fig.savefig(RES / "fig_expA.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("plot failed:", e)
    print(f"done in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
