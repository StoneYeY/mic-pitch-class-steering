"""Job 005 (v4): validate the FIXED w2s.metrics.ClapScorer on a REAL generated clip.
A piano clip should score much higher against its own piano prompt than against metal/dog."""
import glob
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from w2s import data, metrics  # noqa: E402

clap = metrics.ClapScorer()
print("ClapScorer device:", clap.device)

cands = sorted(glob.glob("runs/004_expC_burst/base-p00-*.wav")) or sorted(glob.glob("runs/**/base-*.wav", recursive=True))
print("clip:", cands[0] if cands else "NONE")
y, sr = sf.read(cands[0])
ae = clap.audio_embed([(y.T if y.ndim > 1 else y, sr)])
prompts = [data.DEV_PROMPTS[0], "aggressive heavy metal electric guitar solo", "a barking dog in a park", data.TEST_PROMPTS[0]]
te = clap.text_embed(prompts)
for i, p in enumerate(prompts):
    print(f"  cos = {float(clap.score(ae, te[i:i+1])[0]):+.4f}   <- {p[:55]}")
print("audio emb shape:", ae.shape, "norm:", float(np.linalg.norm(ae)))
# FAD sanity: distance between two disjoint random-embedding halves should be ~0 for identical dist
print("DONE (expect the first/last piano prompts highest, metal/dog lowest)")
