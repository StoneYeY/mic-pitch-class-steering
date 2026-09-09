"""Collect a curated demo/listening-test set from the Exp B wavs and transcode to mp3
(small enough to sync back via git). Emits results/demo/manifest.json + mp3s.

For each (prompt, melody) in a curated list, copy the SAO / MLSP-fixed / RAPG-cal(dC) clips.
Also renders the target melody as a short reference tone mp3."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from w2s import data  # noqa: E402

RUNSB = Path(os.environ.get("EXPB_RUNS", "runs/009_expB_full"))
OUT = Path(os.environ.get("W2S_RESULTS", "results/demo")); (OUT / "audio").mkdir(parents=True, exist_ok=True)
# curated (prompt_id, melody) pairs spanning styles + melody types
PAIRS = [(0, "ascending"), (2, "alternating"), (4, "pedal"), (6, "zigzag"), (10, "descending"), (12, "ascending")]
CONDS = {"sao": "SAO (unguided)", "mlsp": "MLSP-fixed", "rapg_cal_dc": "RAPG-cal(dC)"}


def to_mp3(src: Path, dst: Path):
    """Compact browser-safe copy: mono 22.05 kHz 16-bit WAV (no codec dependency)."""
    try:
        import librosa
        y, s = sf.read(str(src)); y = y.mean(1) if y.ndim > 1 else y
        y = librosa.resample(y.astype(np.float32), orig_sr=s, target_sr=22050)
        sf.write(str(dst), y, 22050, subtype="PCM_16")
        return True
    except Exception as e:  # noqa: BLE001
        print("transcode failed:", e); return False


def target_tone(mel: data.Melody, sr=44100) -> np.ndarray:
    beat = 60.0 / mel.tempo
    y = np.zeros(int(sr * mel.duration_s), np.float32)
    for i, n in enumerate(mel.notes):
        f = 440 * 2 ** ((n - 69) / 12)
        a, b = int(i * beat * sr), int((i + 1) * beat * sr)
        t = np.arange(b - a) / sr
        env = np.minimum(1, np.minimum(t * 20, (b - a - np.arange(b - a)) / sr * 20))
        y[a:b] = 0.25 * np.sin(2 * np.pi * f * t) * env
    return y


man = []
for pid, mel in PAIRS:
    m = data.MELODIES[mel]
    tone = target_tone(m)
    tref = OUT / "audio" / f"target_{mel}.wav"; sf.write(str(tref), tone, 44100)
    to_mp3(tref, tref.with_suffix(".demo.wav")); tref.unlink(missing_ok=True)
    entry = dict(prompt_id=pid, prompt=data.TEST_PROMPTS[pid], melody=mel,
                 target=f"audio/target_{mel}.demo.wav", clips={})
    tid = f"test-p{pid:02d}-{mel}-s0"
    for cond in CONDS:
        src = RUNSB / f"{tid}__{cond}.wav"
        if not src.exists():
            print("missing", src); continue
        dst = OUT / "audio" / f"{tid}__{cond}.demo.wav"
        if to_mp3(src, dst):
            entry["clips"][cond] = f"audio/{tid}__{cond}.demo.wav"
    man.append(entry)
    print(f"{tid}: {list(entry['clips'])}")
json.dump({"conditions": CONDS, "items": man}, open(OUT / "manifest.json", "w"), indent=1)
print(f"wrote {len(man)} items, {len(list((OUT/'audio').glob('*.demo.wav')))} demo wavs")
