"""Job 006: definitive CLAP check (fresh filename to avoid stale-code races).
Prints a content marker, tests audio-audio cosines (space sanity, no text) and audio-text."""
import glob
import hashlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa
from transformers import ClapModel, ClapProcessor

print("MARKER clap_check v1  md5(self)=", hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:8])
sr = 48000
model = ClapModel.from_pretrained("laion/larger_clap_music").eval().to("cuda")
proc = ClapProcessor.from_pretrained("laion/larger_clap_music")

def aemb(y, s):
    y = y.mean(1) if y.ndim > 1 else y
    y = librosa.resample(y.astype(np.float32), orig_sr=s, target_sr=sr)
    inp = {k: v.to("cuda") for k, v in proc(audio=[y], sampling_rate=sr, return_tensors="pt", padding=True).items()}
    inp = {k: inp[k] for k in inp if k in ("input_features", "is_longer")}
    with torch.no_grad():
        e = model.get_audio_features(**inp).pooler_output
    return torch.nn.functional.normalize(e, dim=-1)

def temb(prompts):
    inp = {k: v.to("cuda") for k, v in proc(text=prompts, return_tensors="pt", padding=True).items()}
    inp = {k: inp[k] for k in inp if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        e = model.get_text_features(**inp).pooler_output
    return torch.nn.functional.normalize(e, dim=-1)

# grab a few distinct real clips
wavs = sorted(glob.glob("runs/004_expC_burst/base-p0*-s0.wav"))[:4]
print("clips:", [Path(w).name for w in wavs])
embs = [aemb(*sf.read(w)) for w in wavs]
print("audio-audio cosine matrix (distinct piano clips, expect 0.3-0.8):")
for i in range(len(embs)):
    print("  " + " ".join(f"{float((embs[i]*embs[j]).sum()):+.3f}" for j in range(len(embs))))
prompts = ["a gentle solo piano piece", "aggressive heavy metal guitar", "a barking dog", "white noise hiss"]
t = temb(prompts)
print("audio[0] (piano) vs texts:")
for i, p in enumerate(prompts):
    print(f"  {float((embs[0]*t[i:i+1]).sum()):+.4f}  {p}")
print("text-text cosine piano-vs-metal:", float((t[0:1]*t[1:2]).sum()))
print("DONE")
