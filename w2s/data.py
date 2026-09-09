"""Prompt / melody sets for the sprint, with a dev/test split, and frame-level targets.

* DEV prompts (5) are used ONLY for trajectory analysis (Exp A), steering
  sensitivity (Exp C) and for calibrating RAPG (profile w(t), eta, r_bar, lambda).
* TEST prompts (15) are used ONLY for the final table (Exp B).
* Melodies are pitch-class only, F major, 120 BPM, one note per beat.
  The first three reproduce the MLSP paper; the last two are new.

NOTE: if the original MLSP repo ships its own 9 prompts / 3 melodies, replace the
corresponding entries here (keep the dev set disjoint from them).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np

# --------------------------------------------------------------------------- prompts
DEV_PROMPTS: list[str] = [                       # new, never used in the final table
    "A gentle solo piano lullaby, slow and tender",
    "Upbeat ragtime piano, lively and playful",
    "Dark cinematic piano with deep reverb, suspenseful",
    "Jazz piano trio, relaxed swing feel, brushed drums",
    "Minimalist piano with soft repeating patterns, ambient",
]

# The 9 MLSP prompts (musicpref "piano" prompts, scripts/03_run_inference.py) + 6 new ones.
TEST_PROMPTS: list[str] = [
    "Fast-paced Western Classical music with lively violin harmony, electric cello, piano and string accompaniments that are colourful, well-layered, dense, rich, full, pleasant, cheerful, merry and elegant.",
    "An alternative/indie song with groovy bass, punchy drums, piano chords, and synth melodies that create an addictive and retro sound.",
    "Classical music piece with a gentle piano tune and a theremin playing the main melody, creating a unique and heart-touching atmosphere suitable for an animation movie/TV series.",
    "Upbeat indie rock instrumental with piano melody, simple percussion, distortion guitar, and bass.",
    "A slow and captivating piano piece that evokes sombre and reflective emotions with simple yet effective motifs.",
    "Dramatic contemporary classical piano piece with accentuated playing style suitable for documentary or mystery/horror video game soundtrack.",
    "A piano arpeggio melody with reverb and environmental sounds, suitable for amateur video intros or outros.",
    "Relaxing instrumental with piano and French horn, featuring an ascending progression and arpeggiated chords.",
    "A melodic and emotional duet with piano, electric guitar, drums, and synthesizer arrangements that could be suitable for children.",
    "A calm classical piano piece in the style of a nocturne",
    "Cozy jazz piano in a small club, late night",
    "Contemporary neo-classical piano, introspective",
    "Melancholic piano ballad, rainy day mood",
    "Lo-fi piano loop with vinyl crackle, chill",
    "Grand piano in a concert hall, dramatic and powerful",
]
N_MLSP_PROMPTS = 9

# --------------------------------------------------------------------------- melodies
@dataclass(frozen=True)
class Melody:
    name: str
    notes: tuple[int, ...]          # MIDI note numbers, one per beat
    tempo: float = 120.0            # BPM
    note_beats: float = 1.0         # duration of each note in beats

    @property
    def duration_s(self) -> float:
        return len(self.notes) * 60.0 / self.tempo * self.note_beats

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return tuple(n % 12 for n in self.notes)


MELODIES: dict[str, Melody] = {
    # --- exactly the MLSP code's TEST_MELODIES (F natural minor, 120 BPM; the paper text says F major) ---
    "ascending":   Melody("ascending",   (53, 55, 56, 58, 60, 61, 63, 65)),      # F3 G3 Ab3 Bb3 C4 Db4 Eb4 F4
    "descending":  Melody("descending",  (65, 63, 61, 60, 58, 56, 55, 53)),      # F4 .. F3
    "alternating": Melody("alternating", (53, 56, 60, 65, 60, 56, 53)),          # F3 Ab3 C4 F4 C4 Ab3 F3 (7 notes)
    # --- new for ICASSP ---
    "pedal":       Melody("pedal",       (60, 60, 60, 60, 65, 65, 65, 65)),      # repeated-note pedal: C4 x4, F4 x4
    "zigzag":      Melody("zigzag",      (53, 60, 56, 63, 58, 65, 60, 68)),      # leap-heavy: F3 C4 Ab3 Eb4 Bb3 F4 C4 Ab4
}
MLSP_MELODIES = ("ascending", "descending", "alternating")
SEEDS_3 = (0, 1, 2)
SEEDS_5 = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- targets
def pitch_class_target(melody: Melody, n_frames: int, clip_seconds: float = 5.0,
                       start_s: float = 0.0) -> np.ndarray:
    """Binary (12, T) target on a grid of `n_frames` frames spanning `clip_seconds` of audio
    (frame rate = n_frames / clip_seconds, so it matches whatever the VAE latent length is).
    Frames outside the melody are all-zero (= 'not target-active')."""
    y = np.zeros((12, n_frames), dtype=np.float32)
    frame_rate = n_frames / clip_seconds
    beat = 60.0 / melody.tempo * melody.note_beats
    for i, n in enumerate(melody.notes):
        t0 = start_s + i * beat
        t1 = t0 + beat
        f0, f1 = int(np.floor(t0 * frame_rate)), int(np.floor(t1 * frame_rate))
        y[n % 12, max(f0, 0):min(f1, n_frames)] = 1.0
    return y


def active_frames(target: np.ndarray) -> np.ndarray:
    """Boolean (T,) mask of frames where the target has at least one active pitch class."""
    return target.sum(axis=0) > 0


# --------------------------------------------------------------------------- trial grids
@dataclass(frozen=True)
class Trial:
    split: str          # 'dev' | 'test'
    prompt_id: int      # index into DEV_PROMPTS / TEST_PROMPTS
    melody: str
    seed: int

    @property
    def prompt(self) -> str:
        return (DEV_PROMPTS if self.split == "dev" else TEST_PROMPTS)[self.prompt_id]

    @property
    def tid(self) -> str:
        return f"{self.split}-p{self.prompt_id:02d}-{self.melody}-s{self.seed}"


def trial_grid(split: str, melodies: Iterable[str] = tuple(MELODIES), seeds: Iterable[int] = SEEDS_3,
               prompt_ids: Iterable[int] | None = None) -> list[Trial]:
    prompts = DEV_PROMPTS if split == "dev" else TEST_PROMPTS
    pids = list(prompt_ids) if prompt_ids is not None else list(range(len(prompts)))
    return [Trial(split, p, m, s) for p, m, s in product(pids, melodies, seeds)]


if __name__ == "__main__":
    for m in MELODIES.values():
        print(f"{m.name:10s} {m.notes} pcs={m.pitch_classes} dur={m.duration_s:.1f}s")
    y = pitch_class_target(MELODIES["ascending"], 215)
    print("target shape", y.shape, "active frames", int(active_frames(y).sum()))
    print("dev trials (5x5x2):", len(trial_grid("dev", seeds=(0, 1))))
    print("test trials (15x5x3):", len(trial_grid("test")))
