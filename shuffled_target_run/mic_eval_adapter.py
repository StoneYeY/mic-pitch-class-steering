"""Adapter exposing MIC's own evaluation logic to shuffled_target_baseline.py.

MELODIES / analyze / build_target / melodic_coherence reproduce, step for step,
`compute_melody_coherence` in scripts/04_evaluate.py (the code that produced the
paper's Table 2 numbers), re-shaped into the signatures the handoff script
expects. The multi-hot (n_frames, 12) target is the one-hot encoding of the
original monophonic per-frame pitch-class target, so np.roll on the pitch-class
axis is exactly a circular rotation of the original target.
"""
from pathlib import Path

import numpy as np
import librosa

FRAME_RATE = 86.0          # 04_evaluate.py: hop_length = int(sr / frame_rate)
TEMPO = 120.0
NOTE_DURATION_BEATS = 1.0

# Copied from scripts/03_run_inference.py / scripts/04_evaluate.py TEST_MELODIES
MELODIES = {
    "ascending":   [53, 55, 56, 58, 60, 61, 63, 65],
    "descending":  [65, 63, 61, 60, 58, 56, 55, 53],
    "alternating": [53, 56, 60, 65, 60, 56, 53],
}

# 04_evaluate.py derives fps from the actual audio duration, which build_target's
# signature doesn't carry. analyze() stashes it here; the runner always calls
# analyze(path) immediately before build_target for the same clip.
_last_duration = None


def analyze(path):
    """pYIN exactly as in compute_melody_coherence; cached next to the wav."""
    global _last_duration
    cache = Path(path).with_suffix(".pyin.npz")
    if cache.exists():
        d = np.load(cache)
        _last_duration = float(d["duration"])
        return d["pc"], d["voiced"]

    y, sr = librosa.load(path, sr=None, mono=True)
    duration = len(y) / sr
    hop_length = int(sr / FRAME_RATE)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length,
    )
    midi = librosa.hz_to_midi(f0)
    voiced = np.isfinite(midi)          # == (pc_gen != -1) in the original
    pc = np.zeros(len(midi), dtype=np.int64)
    pc[voiced] = np.round(midi[voiced]).astype(int) % 12
    np.savez_compressed(cache, pc=pc, voiced=voiced, duration=duration)
    _last_duration = duration
    return pc, voiced


def build_target(midi_notes, n_frames):
    """One-hot encoding of the original pc_target construction."""
    if _last_duration is None:
        raise RuntimeError("analyze() must be called before build_target()")
    audio_duration = _last_duration
    sec_per_beat = 60.0 / TEMPO
    note_sec = NOTE_DURATION_BEATS * sec_per_beat
    fps = n_frames / audio_duration
    target = np.zeros((n_frames, 12), dtype=np.int8)
    t = 0.0
    for note in midi_notes:
        if t >= audio_duration:
            break
        start_frame = int(t * fps)
        end_frame = int((t + note_sec) * fps)
        end_frame = max(end_frame, start_frame + 1)
        end_frame = min(end_frame, n_frames)
        target[start_frame:end_frame, note % 12] = 1
        t += note_sec
    return target


def melodic_coherence(pitch_classes, voiced, target):
    """Original: valid = (pc_target != -1) & (pc_gen != -1);
    coherence = (pc_target[valid] == pc_gen[valid]).mean();
    voiced_coverage = valid.sum() / target_frames.
    """
    active = target.sum(axis=1) > 0
    target_frames = int(active.sum())
    valid = active & voiced
    voiced_frames = int(valid.sum())
    voiced_fraction = voiced_frames / max(1, target_frames)
    if voiced_frames == 0:
        return float("nan"), voiced_fraction
    idx = np.flatnonzero(valid)
    hit = target[idx, pitch_classes[idx]] > 0
    return float(hit.mean()), voiced_fraction
