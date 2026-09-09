"""Job 002: end-to-end smoke test of the corrected, schedule-driven generator.

1 dev prompt x 'ascending' x seed 0, three runs:
  sao        : unguided, trajectory saved  -> probe F1(t) on real trajectory (pseudo-labels from pYIN)
  mlsp_fixed : corrected target alignment, K=15 (steps 20,22,...,48), lambda=0.05
  mlsp_legacy: the MLSP code's placement (target fps = T/audio_length) for comparison
Prints timings, VRAM, R_t per step, coherence (MLSP metric + w2s metric) and chroma cosine.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from w2s import data, metrics, probe_eval  # noqa: E402
from w2s.generate import W2SGenerator  # noqa: E402
from w2s.mlsp_bridge import load_pipeline, load_probe, mlsp_coherence  # noqa: E402
from w2s.schedules import make_schedule  # noqa: E402

RES = Path(os.environ.get("W2S_RESULTS", "results/smoke")); RES.mkdir(parents=True, exist_ok=True)
RUNS = Path(os.environ.get("W2S_RUNS", "runs/smoke")); RUNS.mkdir(parents=True, exist_ok=True)


def main():
    import diffusers, transformers
    print("torch", torch.__version__, "diffusers", diffusers.__version__, "transformers", transformers.__version__)
    print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    t0 = time.time()
    pipe = load_pipeline()
    probe = load_probe()
    print(f"loaded pipeline+probe in {time.time()-t0:.1f}s; probe params={sum(p.numel() for p in probe.parameters()):,}")
    gen = W2SGenerator(pipe, probe)
    print(f"sr={gen.sr} hop={gen.hop} fps={gen.fps:.3f} T={gen.T} audible_frames={int(np.ceil(5*gen.fps))}")

    prompt = data.DEV_PROMPTS[0]
    mel = data.MELODIES["ascending"]
    seed = 0
    runs = {
        "sao": dict(schedule=make_schedule("sao"), save_trajectory=True),
        "mlsp_fixed": dict(schedule=make_schedule("mlsp", lam=0.05), save_trajectory=True),
        "mlsp_legacy": dict(schedule=make_schedule("mlsp", lam=0.05), target_fps=gen.T / 5.0, loss_region="all"),
    }
    summary = {}
    for name, kw in runs.items():
        r = gen.run(prompt, seed, mel, verbose=(name != "sao"), **kw)
        wav = RUNS / f"{name}.wav"
        sf.write(str(wav), r.audio, r.sr)
        coh_mlsp = mlsp_coherence(wav, list(mel.notes))
        pc, voiced = metrics.pyin_pitch_classes(r.audio.T, r.sr)                 # hop 2048 -> latent fps
        tgt_aud = r.target[:, :r.n_audible]
        coh_w2s = metrics.melodic_coherence(pc[:r.n_audible], voiced[:r.n_audible], tgt_aud)
        chroma = metrics.chroma_cosine(r.audio.T, r.sr, tgt_aud)
        Rs = [x["R"] for x in r.log]
        summary[name] = dict(runtime_s=round(r.runtime_s, 2), peak_vram_gb=round(r.peak_vram_gb, 2),
                             n_updates=r.n_updates, coherence_mlsp=coh_mlsp["coherence"],
                             voiced_cov=coh_mlsp["voiced_coverage"], coherence_w2s=coh_w2s, chroma_cos=chroma,
                             R_first=Rs[0], R_mid=Rs[len(Rs) // 2], R_last=Rs[-1])
        print(f"\n[{name}] {json.dumps(summary[name])}")
        print("  R_t:", " ".join(f"{x:.2f}" for x in Rs))
        print("  sigma:", " ".join(f"{x['sigma']:.2f}" for x in r.log[::5]))
        json.dump(r.log, open(RES / f"{name}_log.json", "w"))
        if r.trajectory is not None:
            np.savez_compressed(RUNS / f"{name}_traj.npz", z=r.trajectory, target=r.target, n_audible=r.n_audible,
                                sigma=np.array([x["sigma"] for x in r.log]))
            # probe reliability along the real trajectory vs pseudo-labels from the generated audio
            y_ref, mask = metrics.pseudo_label(pc, voiced, r.n_audible)
            y_full = np.zeros((12, gen.T), np.float32); y_full[:, :r.n_audible] = y_ref
            m_full = np.zeros(gen.T, bool); m_full[:r.n_audible] = mask
            rows = probe_eval.evaluate_trajectory(gen.probe_fn, r.trajectory, y_full, m_full, device=str(gen.device))
            print("  step  f1   posrec  bce   R_ent  R_mar  agree")
            for row in rows[::5] + [rows[-1]]:
                print(f"  {row['row']:4d} {row['f1']:.3f} {row['pos_recall']:.3f} {row['bce']:.3f} "
                      f"{row['R_entropy']:.3f} {row['R_margin']:.3f} {row['agree_final']:.3f}")
            json.dump(rows, open(RES / f"{name}_probe_eval.json", "w"))
    json.dump(summary, open(RES / "summary.json", "w"), indent=2)
    print("\nSUMMARY", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
