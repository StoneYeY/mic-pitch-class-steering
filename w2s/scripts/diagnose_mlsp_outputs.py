"""Job 001: does the MLSP guidance follow the melody over time, or push the whole clip
toward the first note's pitch class (F)?  Runs on the 189 wavs the MLSP paper was
evaluated on (9 prompts x 3 melodies x 7 conditions) if they are present.

Per clip: pYIN (same settings as scripts/04_evaluate.py) -> per-frame pitch class ->
  coh_true     : coherence vs the true time-varying target      (paper metric)
  coh_first    : coherence vs a target that is the FIRST note's pc for the whole melody span
  coh_best_const: best constant-pc target (upper bound of "steer everything to one pc")
  frac_F       : fraction of voiced frames (whole clip) whose pc == F (5)
  slot_acc     : per note-slot accuracy vs true target (list)
Outputs: results csv + summary json (+ per-method/melody means).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from w2s.mlsp_bridge import MLSP_MELODIES  # noqa: E402

METHODS = ["baseline", "combined", "probe_low", "probe_mid", "probe_high", "probe_strong", "probe", "itc"]


def pyin_pc(path: str, frame_rate: float = 86.0):
    import librosa
    y, sr = librosa.load(path, sr=None, mono=True)
    hop = int(sr / frame_rate)
    f0, vflag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr, hop_length=hop)
    midi = librosa.hz_to_midi(f0)
    pc = np.full(len(midi), -1, dtype=int)
    ok = np.isfinite(midi)
    pc[ok] = np.round(midi[ok]).astype(int) % 12
    return pc, len(y) / sr


def slot_targets(notes, n_frames, duration, tempo=120.0):
    fps = n_frames / duration
    beat = 60.0 / tempo
    tgt = np.full(n_frames, -1, dtype=int)
    slot = np.full(n_frames, -1, dtype=int)
    t = 0.0
    for i, n in enumerate(notes):
        if t >= duration:
            break
        a, b = int(t * fps), int((t + beat) * fps)
        b = min(max(b, a + 1), n_frames)
        tgt[a:b] = n % 12
        slot[a:b] = i
        t += beat
    return tgt, slot


def analyse(path: Path, notes):
    pc, dur = pyin_pc(str(path))
    tgt, slot = slot_targets(notes, len(pc), dur)
    active = tgt >= 0
    valid = active & (pc >= 0)
    cov = valid.sum() / max(1, active.sum())
    coh_true = float((pc[valid] == tgt[valid]).mean()) if valid.any() else np.nan
    first = notes[0] % 12
    coh_first = float((pc[valid] == first).mean()) if valid.any() else np.nan
    consts = [float((pc[valid] == k).mean()) for k in range(12)] if valid.any() else [np.nan] * 12
    voiced = pc >= 0
    hist = np.bincount(pc[voiced], minlength=12) / max(1, voiced.sum())
    slot_acc = []
    for i in range(len(notes)):
        m = valid & (slot == i)
        slot_acc.append(float((pc[m] == tgt[m]).mean()) if m.any() else np.nan)
    return dict(coh_true=coh_true, coh_first=coh_first, coh_best_const=float(np.nanmax(consts)),
                best_const_pc=int(np.nanargmax(consts)), frac_F=float(hist[5]), voiced_cov=float(cov),
                hist=hist.round(3).tolist(), slot_acc=slot_acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default=str(Path.home() / "Desktop" / "MIC" / "outputs" / "generated_audio"))
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    adir = Path(args.audio_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    wavs = sorted(adir.glob("*.wav"))
    print(f"audio dir: {adir}  ({len(wavs)} wavs)")
    if not wavs:
        print("nothing to analyse (directory missing or empty) -> skipping")
        return
    rows = []
    for w in wavs:
        name = w.name
        method = next((m for m in METHODS if f"_{m}.wav" in name), "unknown")
        melody = next((k for k in MLSP_MELODIES if k in name), None)
        pm = re.search(r"prompt(\d+)", name)
        if melody is None or pm is None:
            continue
        try:
            r = analyse(w, MLSP_MELODIES[melody])
        except Exception as e:  # noqa: BLE001
            print("  error", name, e); continue
        r.update(file=name, method=method, melody=melody, prompt=int(pm.group(1)))
        rows.append(r)
        print(f"{name:55s} coh_true={r['coh_true']:.3f} coh_first={r['coh_first']:.3f} "
              f"best_const={r['coh_best_const']:.3f}(pc{r['best_const_pc']}) frac_F={r['frac_F']:.3f}")
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "mlsp_output_diagnosis.csv", index=False)
    summ = {}
    print("\n=== per method (mean over clips) ===")
    print(f"{'method':14s} {'n':>3s} {'coh_true':>9s} {'coh_first':>10s} {'best_const':>11s} {'frac_F':>7s}")
    for m, g in df.groupby("method"):
        s = dict(n=int(len(g)), coh_true=float(g.coh_true.mean()), coh_first=float(g.coh_first.mean()),
                 coh_best_const=float(g.coh_best_const.mean()), frac_F=float(g.frac_F.mean()),
                 slot_acc_mean=np.nanmean(np.array([x + [np.nan] * (8 - len(x)) for x in g.slot_acc]), axis=0).round(3).tolist())
        summ[m] = s
        print(f"{m:14s} {s['n']:3d} {s['coh_true']:9.3f} {s['coh_first']:10.3f} {s['coh_best_const']:11.3f} {s['frac_F']:7.3f}")
    print("\n=== per method x melody: coh_true / coh_first / frac_F ===")
    for (m, mel), g in df.groupby(["method", "melody"]):
        print(f"{m:14s} {mel:12s} {g.coh_true.mean():.3f} / {g.coh_first.mean():.3f} / {g.frac_F.mean():.3f}")
        summ.setdefault("by_melody", {})[f"{m}/{mel}"] = dict(coh_true=float(g.coh_true.mean()),
                                                              coh_first=float(g.coh_first.mean()),
                                                              frac_F=float(g.frac_F.mean()))
    print("\n=== per-slot accuracy (probe_mid vs baseline): does the audio follow the melody in time? ===")
    for m in ("baseline", "probe_mid"):
        if m in summ:
            print(f"{m:12s} slot_acc = {summ[m]['slot_acc_mean']}")
    json.dump(summ, open(out / "summary.json", "w"), indent=2)
    print("\nInterpretation: if probe_* clips have coh_first ~ coh_true and frac_F >> baseline, guidance mostly\n"
          "pushed the whole clip toward F (the first note), consistent with the target time-axis bug.")


if __name__ == "__main__":
    main()
