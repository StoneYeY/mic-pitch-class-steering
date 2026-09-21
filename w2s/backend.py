"""Pick the generation backend from the environment.

  W2S_BACKEND=sao (default)  Stable Audio Open 1.0 via diffusers + the MLSP probe (w2s.generate.W2SGenerator)
  W2S_BACKEND=sa3            Stable Audio 3 (medium-base) via stable-audio-tools + the SA3 probe (w2s.sa3.SA3Generator)
Both expose: run(prompt, seed, melody, schedule, ...) -> RunResult, make_target, probe_fn, probe, fps, T, sr, device.
"""
from __future__ import annotations

import os

BACKEND = os.environ.get("W2S_BACKEND", "sao").lower()


ALPHA = float(os.environ.get("W2S_ALPHA", "0.05"))           # clamp of the update, alpha * rms(z)
RANDOM_GRAD = os.environ.get("W2S_RANDOM_GRAD", "0") == "1"  # random-direction control (same update norm)


def make_generator(device: str = "cuda"):
    if BACKEND == "sa3":
        from .sa3 import SA3Generator, load_sa3, load_sa3_probe
        model, cfg = load_sa3(device)
        probe = load_sa3_probe(device=device)
        gen = SA3Generator(model, cfg, probe, device=device, alpha=ALPHA)
    else:
        from .generate import W2SGenerator
        from .mlsp_bridge import load_pipeline, load_probe
        pipe, probe = load_pipeline(), load_probe()
        gen = W2SGenerator(pipe, probe, alpha=ALPHA)
    gen.random_grad = RANDOM_GRAD
    if RANDOM_GRAD or ALPHA != 0.05:
        print(f"[backend] alpha={ALPHA} random_grad={RANDOM_GRAD}")
    return gen
