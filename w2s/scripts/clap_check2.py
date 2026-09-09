"""Job 007: find a working CLAP. transformers 5.7 collapses larger_clap_music (all embeds ~equal).
Test (A) transformers laion/clap-htsat-unfused, (B) the laion_clap package. A working one gives
text-text(piano,metal) well below 0.9 and varied audio-audio cosines."""
import glob
import numpy as np
import soundfile as sf
import torch
import librosa
from pathlib import Path

wavs = sorted(glob.glob("runs/004_expC_burst/base-p0*-s0.wav"))[:4]
prompts = ["a gentle solo piano piece", "aggressive heavy metal guitar solo", "a barking dog"]
def load_audio(w, sr):
    y, s = sf.read(w); y = y.mean(1) if y.ndim > 1 else y
    return librosa.resample(y.astype(np.float32), orig_sr=s, target_sr=sr)

def report(name, A, T):  # A:(n,d) audio, T:(m,d) text, both normalized np
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    print(f"[{name}] audio-audio offdiag mean = {(A@A.T)[~np.eye(len(A),dtype=bool)].mean():.3f} "
          f"(want <0.9); text piano-vs-metal = {float(T[0]@T[1]):.3f} (want <0.9)")
    print(f"   audio[0] vs texts: " + " ".join(f"{float(A[0]@T[i]):+.3f}" for i in range(len(prompts))))

# (A) transformers, different checkpoint
try:
    from transformers import ClapModel, ClapProcessor
    for ckpt in ("laion/clap-htsat-unfused",):
        m = ClapModel.from_pretrained(ckpt).eval().to("cuda"); p = ClapProcessor.from_pretrained(ckpt)
        def ae(y):
            inp = {k: v.to("cuda") for k, v in p(audio=[y], sampling_rate=48000, return_tensors="pt", padding=True).items() if k in ("input_features","is_longer")}
            with torch.no_grad(): return torch.nn.functional.normalize(m.get_audio_features(**inp).pooler_output, dim=-1).cpu().numpy()
        def te(ts):
            inp = {k: v.to("cuda") for k, v in p(text=ts, return_tensors="pt", padding=True).items() if k in ("input_ids","attention_mask")}
            with torch.no_grad(): return torch.nn.functional.normalize(m.get_text_features(**inp).pooler_output, dim=-1).cpu().numpy()
        A = np.concatenate([ae(load_audio(w, 48000)) for w in wavs]); T = te(prompts)
        report(f"transformers {ckpt}", A, T)
except Exception as e:
    print("[A] transformers path failed:", type(e).__name__, e)

# (B) laion_clap package
try:
    import laion_clap
    m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    m.load_ckpt()  # downloads 630k-best if absent
    A = m.get_audio_embedding_from_data(x=np.stack([load_audio(w, 48000) for w in wavs]), use_tensor=False)
    T = m.get_text_embedding(prompts, use_tensor=False)
    report("laion_clap HTSAT-base", np.asarray(A), np.asarray(T))
except Exception as e:
    import traceback; print("[B] laion_clap path failed:", type(e).__name__, e); traceback.print_exc()
print("DONE")
