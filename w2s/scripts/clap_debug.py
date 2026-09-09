"""Job 005: surface the CLAP failure and find a call signature that works under transformers 5.7."""
import traceback
import numpy as np
import torch

print("torch", torch.__version__)
import transformers
print("transformers", transformers.__version__)

sr = 48000
y = (0.1 * np.random.randn(sr * 5)).astype(np.float32)

from transformers import ClapModel, ClapProcessor
model = ClapModel.from_pretrained("laion/larger_clap_music").eval()
proc = ClapProcessor.from_pretrained("laion/larger_clap_music")
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev)
print("loaded; processor class:", type(proc).__name__)

def try_variant(name, fn):
    try:
        out = fn()
        print(f"[OK] {name}: shape={tuple(out.shape)} finite={bool(np.isfinite(out.detach().cpu().numpy()).all())}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()

def v_audio_kw():
    inp = proc(audios=[y], sampling_rate=sr, return_tensors="pt", padding=True)
    print("   audios= keys:", list(inp.keys()), {k: tuple(v.shape) for k, v in inp.items()})
    with torch.no_grad():
        return model.get_audio_features(**{k: v.to(dev) for k, v in inp.items()})

def v_audio_singular():
    inp = proc(audio=[y], sampling_rate=sr, return_tensors="pt", padding=True)
    print("   audio= keys:", list(inp.keys()), {k: tuple(v.shape) for k, v in inp.items()})
    with torch.no_grad():
        return model.get_audio_features(**{k: v.to(dev) for k, v in inp.items()})

def v_feature_extractor():
    from transformers import AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained("laion/larger_clap_music")
    inp = fe([y], sampling_rate=sr, return_tensors="pt")
    print("   FE keys:", list(inp.keys()), {k: tuple(v.shape) for k, v in inp.items()})
    with torch.no_grad():
        return model.get_audio_features(**{k: v.to(dev) for k, v in inp.items()})

def v_text():
    inp = proc(text=["a piano piece"], return_tensors="pt", padding=True)
    with torch.no_grad():
        return model.get_text_features(**{k: v.to(dev) for k, v in inp.items()})

try_variant("audios= (plural)", v_audio_kw)
try_variant("audio= (singular)", v_audio_singular)
try_variant("AutoFeatureExtractor", v_feature_extractor)
try_variant("text", v_text)
print("DONE")
