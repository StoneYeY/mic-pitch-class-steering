"""Job 005 (v2): find the correct way to get projected CLAP embeddings under transformers 5.7,
and verify the space by checking that a matching prompt scores higher than a mismatched one."""
import inspect
import numpy as np
import torch
import transformers
from transformers import ClapModel, ClapProcessor

print("transformers", transformers.__version__)
sr = 48000
rng = np.random.default_rng(0)
# a crude 440 Hz tone vs noise, just to see relative CLAP scores
t = np.arange(sr * 5) / sr
tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
noise = (0.05 * rng.standard_normal(sr * 5)).astype(np.float32)

model = ClapModel.from_pretrained("laion/larger_clap_music").eval().to("cuda")
proc = ClapProcessor.from_pretrained("laion/larger_clap_music")

print("get_audio_features sig:", str(inspect.signature(model.get_audio_features)))

def audio_emb(y):
    inp = proc(audio=[y], sampling_rate=sr, return_tensors="pt", padding=True)
    inp = {k: v.to("cuda") for k, v in inp.items()}
    with torch.no_grad():
        out = model.get_audio_features(**inp)
    return out

def text_emb(s):
    inp = proc(text=[s], return_tensors="pt", padding=True)
    inp = {k: v.to("cuda") for k, v in inp.items()}
    with torch.no_grad():
        out = model.get_text_features(**inp)
    return out

ao = audio_emb(tone)
print("type:", type(ao).__name__)
if hasattr(ao, "keys"):
    print("attributes:", [k for k in ao.keys()])
    for k in ao.keys():
        v = ao[k]
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}")
for attr in ("pooler_output", "audio_embeds", "last_hidden_state"):
    v = getattr(ao, attr, None)
    if v is not None and hasattr(v, "shape"):
        print(f"  .{attr}: {tuple(v.shape)}")

def as_tensor(o):
    if torch.is_tensor(o):
        return o
    for attr in ("audio_embeds", "text_embeds", "pooler_output"):
        v = getattr(o, attr, None)
        if v is not None:
            return v
    return o[0] if isinstance(o, (tuple, list)) else o

at = as_tensor(ao)
print("chosen audio tensor shape:", tuple(at.shape))
# verify shared space: tone should match "a sine tone / instrument" better than "a barking dog"
a = torch.nn.functional.normalize(as_tensor(audio_emb(tone)), dim=-1)
for prompt in ["a bright piano melody", "a barking dog", "a solo violin note", "white noise static"]:
    tt = torch.nn.functional.normalize(as_tensor(text_emb(prompt)), dim=-1)
    print(f"  cos(tone, '{prompt}') = {float((a*tt).sum()):.4f}")
an = torch.nn.functional.normalize(as_tensor(audio_emb(noise)), dim=-1)
tn = torch.nn.functional.normalize(as_tensor(text_emb('white noise static')), dim=-1)
print(f"  cos(noise,'white noise static') = {float((an*tn).sum()):.4f}")
print("DONE")
