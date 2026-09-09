"""Exp B: guidance-window ablation + RAPG at a matched budget K.

Methods (all K updates unless noted):
  sao, early, mid, late, uniform, mlsp (MLSP-fixed),
  rapg_cal_f1  : Top-K of the dev-set decodability profile F1(t)      (Exp A, variant raw)
  rapg_cal_dc  : Top-K of the dev-set steerability profile dCoh(t)     (Exp C, interpolated)
  rapg_on      : online: guide iff R_t >= eta (eta calibrated on dev R_t so that mean #updates ~ K), cap K
  topk_dc_const: Top-K of dCoh(t) with CONSTANT lambda (isolates 'when' from 'how strong')
Strength: fixed methods use lambda; RAPG uses lambda_t = lambda * R_t / r_bar (r_bar from dev).

Env:  SPLIT=dev|test  SEEDS=0,1,2  MELODIES=...  PROMPTS=0,1,...  METHODS=comma list  K=15  LAMBDA=0.05
      EXPA_DIR=results/003_expA_trajectories  EXPC_DIR=results/004_expC_burst
      REF_DIR=~/Desktop/MIC/outputs/reference_audio (MAESTRO 5 s clips, for FAD)
Outputs: results/<job>/per_run.csv (resumable), schedules.json, clap_emb.npz, logs/<run>.json,
         fad.csv, table.csv (via aggregate at the end).  Wavs in runs/<job>/.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from w2s import data, metrics, schedules as S  # noqa: E402
from w2s.evalclip import ClipEvaluator  # noqa: E402
from w2s.generate import W2SGenerator  # noqa: E402
from w2s.mlsp_bridge import load_pipeline, load_probe  # noqa: E402

RES = Path(os.environ.get("W2S_RESULTS", "results/expB")); RES.mkdir(parents=True, exist_ok=True)
RUNS = Path(os.environ.get("W2S_RUNS", "runs/expB")); RUNS.mkdir(parents=True, exist_ok=True)
(RES / "logs").mkdir(exist_ok=True)
SPLIT = os.environ.get("SPLIT", "test")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
MELODIES = os.environ.get("MELODIES", ",".join(data.MELODIES)).split(",")
PROMPTS = [int(x) for x in os.environ["PROMPTS"].split(",")] if os.environ.get("PROMPTS") else None
METHODS = os.environ.get("METHODS", "sao,early,mid,late,uniform,mlsp,rapg_cal_f1,rapg_cal_dc,rapg_on,topk_dc_const").split(",")
K = int(os.environ.get("K", "15"))
LAM = float(os.environ.get("LAMBDA", "0.05"))
LAM_RAPG = float(os.environ.get("LAMBDA_RAPG", str(LAM)))
EXPA = Path(os.environ.get("EXPA_DIR", "results/003_expA_trajectories"))
EXPC = Path(os.environ.get("EXPC_DIR", "results/004_expC_burst"))
REF_DIR = Path(os.path.expanduser(os.environ.get("REF_DIR", "~/Desktop/MIC/outputs/reference_audio")))
STEPS = 50


def build_schedules() -> dict[str, S.GuidanceSchedule]:
    """Calibrate everything from dev-set result files; fall back to sensible defaults with a warning."""
    prof_f1 = prof_dc = None
    R_dev = None
    if (EXPA / "summary.csv").exists():
        s = pd.read_csv(EXPA / "summary.csv"); s = s[s.variant == "raw"].sort_values("row")
        prof_f1 = s.f1.values
        ps = pd.read_csv(EXPA / "per_step.csv"); ps = ps[ps.variant == "raw"]
        R_dev = ps.pivot(index="run", columns="row", values="R_entropy").values
    else:
        print(f"[expB] WARNING: {EXPA}/summary.csv missing -> rapg_cal_f1 / rapg_on use fallbacks")
    if (EXPC / "by_position.csv").exists():
        bp = pd.read_csv(EXPC / "by_position.csv").sort_values("position")
        prof_dc = S.interpolate_profile(bp.position.values, bp.d_coherence_mlsp.values, STEPS, smooth=3)
    else:
        print(f"[expB] WARNING: {EXPC}/by_position.csv missing -> rapg_cal_dc uses fallback")
    sch: dict[str, S.GuidanceSchedule] = {}
    for m in METHODS:
        if m in ("sao", "early", "mid", "late", "uniform", "mlsp"):
            sch[m] = S.make_schedule(m, STEPS, K, LAM)
        elif m == "rapg_cal_f1":
            prof = prof_f1 if prof_f1 is not None else np.linspace(0, 1, STEPS)
            steps = S.topk_steps(prof, K, tie="earliest")
            r_bar = S.mean_reliability_over(R_dev, steps) if R_dev is not None else 1.0
            sch[m] = S.make_schedule("rapg_cal", STEPS, K, LAM_RAPG, profile=prof, r_bar=r_bar, name="RAPG-cal(F1)")
        elif m == "rapg_cal_dc":
            prof = prof_dc if prof_dc is not None else np.exp(-((np.arange(STEPS) - 25) / 10) ** 2)
            steps = S.topk_steps(prof, K, tie="earliest")
            r_bar = S.mean_reliability_over(R_dev, steps) if R_dev is not None else 1.0
            sch[m] = S.make_schedule("rapg_cal", STEPS, K, LAM_RAPG, profile=prof, r_bar=r_bar, name="RAPG-cal(dC)")
        elif m == "topk_dc_const":
            prof = prof_dc if prof_dc is not None else np.exp(-((np.arange(STEPS) - 25) / 10) ** 2)
            sch[m] = S.make_schedule("fixed_adaptive", STEPS, K, LAM, profile=prof, name="TopK(dC)-const")
        elif m == "rapg_on":
            if R_dev is not None:
                eta = float(os.environ.get("ETA", S.calibrate_eta(R_dev, K)))
                r_bar = S.mean_reliability_over(R_dev, eta=eta)
            else:
                eta, r_bar = float(os.environ.get("ETA", "0.5")), 1.0
            sch[m] = S.make_schedule("rapg_on", STEPS, K, LAM_RAPG, eta=eta, r_bar=r_bar)
        else:
            raise ValueError(m)
    print(S.describe(sch.values(), STEPS))
    json.dump({k: v.to_dict() for k, v in sch.items()}, open(RES / "schedules.json", "w"), indent=1)
    return sch


def main():
    sch = build_schedules()
    pipe, probe = load_pipeline(), load_probe()
    gen = W2SGenerator(pipe, probe)
    ev = ClipEvaluator(use_clap=True, device=str(gen.device))
    trials = data.trial_grid(SPLIT, melodies=MELODIES, seeds=SEEDS, prompt_ids=PROMPTS)
    csv = RES / "per_run.csv"
    rows = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    done = {(r["trial"], r["method"]) for r in rows}
    embs = dict(np.load(RES / "clap_emb.npz")) if (RES / "clap_emb.npz").exists() else {}
    print(f"split={SPLIT} trials={len(trials)} methods={list(sch)} -> {len(trials)*len(sch)} runs ({len(done)} done)")
    t0, n_new = time.time(), 0
    for tr in trials:
        mel = data.MELODIES[tr.melody]
        tgt_full = gen.make_target(mel, 5.0)[0][0].T.cpu().numpy()
        n_aud = int(np.ceil(5 * gen.fps)); tgt_aud = tgt_full[:, :n_aud]
        for m, s in sch.items():
            if (tr.tid, m) in done:
                continue
            wav = RUNS / f"{tr.tid}__{m}.wav"
            r = gen.run(tr.prompt, tr.seed, mel, s)
            sf.write(str(wav), r.audio, r.sr)
            e = ev.evaluate(r.audio, gen.sr, mel, tgt_aud, n_aud, tr.prompt, wav)
            if "_clap_emb" in e:
                embs[f"{tr.tid}__{m}"] = e["_clap_emb"]
            guided_steps = [x["step"] for x in r.log if x["guided"]]
            rows.append(dict(trial=tr.tid, method=m, split=tr.split, prompt_id=tr.prompt_id, melody=tr.melody, seed=tr.seed,
                             coherence_mlsp=e["coherence_mlsp"], voiced_cov=e["voiced_cov_mlsp"], coherence_w2s=e["coherence_w2s"],
                             chroma_cos=e["chroma_cos"], clap=e["clap"], coherence_firstnote=e["coherence_firstnote"],
                             n_updates=r.n_updates, first_step=(guided_steps[0] if guided_steps else -1),
                             last_step=(guided_steps[-1] if guided_steps else -1),
                             mean_lam=float(np.mean([x["lam"] for x in r.log if x["guided"]]) if guided_steps else 0.0),
                             runtime_s=r.runtime_s, peak_vram_gb=r.peak_vram_gb))
            json.dump(r.log, open(RES / "logs" / f"{tr.tid}__{m}.json", "w"))
            done.add((tr.tid, m)); n_new += 1
            if n_new % 20 == 0:
                pd.DataFrame(rows).to_csv(csv, index=False); np.savez_compressed(RES / "clap_emb.npz", **embs)
                el = time.time() - t0
                print(f"  {n_new} runs ({el:.0f}s, {el/n_new:.1f}s/run)  last: {tr.tid} {m} coh={e['coherence_mlsp']:.3f} clap={e['clap']:.3f} upd={r.n_updates}")
    df = pd.DataFrame(rows); df.to_csv(csv, index=False); np.savez_compressed(RES / "clap_emb.npz", **embs)
    # FAD (CLAP-music embeddings) vs MAESTRO reference and vs the SAO set
    try:
        if ev.clap is not None and REF_DIR.exists():
            ref_npy = RUNS.parent / "ref_clap_emb.npy"
            if ref_npy.exists():
                ref = np.load(ref_npy)
            else:
                import librosa
                files = sorted(REF_DIR.glob("*.wav"))[:600]
                auds = []
                for f in files:
                    y, sr = librosa.load(str(f), sr=None, mono=True); auds.append((y, sr))
                ref = ev.clap.audio_embed(auds); np.save(ref_npy, ref)
                print(f"reference embeddings: {ref.shape} from {len(files)} clips")
            fad_rows = []
            sao_emb = np.stack([v for k, v in embs.items() if k.endswith("__sao")]) if any(k.endswith("__sao") for k in embs) else None
            for m in sch:
                E = np.stack([v for k, v in embs.items() if k.endswith(f"__{m}")])
                fad_rows.append(dict(method=m, n=len(E), fad_clap_maestro=metrics.frechet_distance(E, ref),
                                     fad_clap_sao=(metrics.frechet_distance(E, sao_emb) if sao_emb is not None and m != "sao" else 0.0)))
            pd.DataFrame(fad_rows).to_csv(RES / "fad.csv", index=False); print(pd.DataFrame(fad_rows).round(3).to_string())
        else:
            print(f"FAD skipped (clap={ev.clap is not None}, ref_dir exists={REF_DIR.exists()})")
    except Exception as ex:  # noqa: BLE001
        print("FAD failed:", ex)
    # quick table
    from w2s.scripts.aggregate_expB import aggregate
    aggregate(RES)
    print(f"done: {n_new} new runs in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
