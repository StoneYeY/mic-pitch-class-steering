"""Encode MAESTRO clips with the Stable Audio 3 autoencoder and build a pitch-class probe dataset
(the MLSP recipe: 10-s clips, frame labels from MIDI at the latent frame rate, split by piece).

  SA3_MAESTRO=<dir with wav+midi pairs>  SA3_MAX_FILES=120  SA3_CLIPS_PER_FILE=6  SA3_CLIP_S=10
  W2S_RESULTS=results/<job>  -> writes data/sa3_probe/{train,val}.pt (gitignored) + results/<job>/dataset.json
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils import midi_to_pitch_class  # noqa: E402
from w2s.sa3 import load_sa3  # noqa: E402

RES = Path(os.environ.get("W2S_RESULTS", "results/sa3_encode")); RES.mkdir(parents=True, exist_ok=True)
OUT = Path(os.environ.get("SA3_DATA", "data/sa3_probe")); OUT.mkdir(parents=True, exist_ok=True)
MAESTRO = os.environ.get("SA3_MAESTRO", os.path.expanduser("~/Desktop/sa_latent_probe/data/maestro_v3_extract/maestro-v3.0.0"))
MAX_FILES = int(os.environ.get("SA3_MAX_FILES", "120"))
CLIPS_PER_FILE = int(os.environ.get("SA3_CLIPS_PER_FILE", "6"))
CLIP_S = float(os.environ.get("SA3_CLIP_S", "10"))
VAL_FRAC = float(os.environ.get("SA3_VAL_FRAC", "0.15"))


def find_pairs(root: str) -> list[tuple[str, str]]:
    pairs = []
    for wav in sorted(glob.glob(f"{root}/**/*.wav", recursive=True)):
        p = Path(wav)
        for ext in (".midi", ".mid"):
            if p.with_suffix(ext).exists():
                pairs.append((wav, str(p.with_suffix(ext)))); break
    return pairs


def main():
    pairs = find_pairs(MAESTRO)
    print(f"{len(pairs)} wav/midi pairs under {MAESTRO}")
    if not pairs:
        sys.exit("no MAESTRO pairs found; set SA3_MAESTRO")
    rng = np.random.default_rng(0)
    rng.shuffle(pairs)
    pairs = pairs[:MAX_FILES]
    n_val = max(1, int(round(VAL_FRAC * len(pairs))))
    split = {"val": pairs[:n_val], "train": pairs[n_val:]}
    model, cfg = load_sa3("cuda")
    sr = cfg["sample_rate"]; hop = cfg["model"]["pretransform"]["config"]["downsampling_ratio"]
    n_samp = int(CLIP_S * sr); n_samp = (n_samp // hop) * hop          # whole latent frames
    fps = sr / hop
    meta = dict(model=os.environ.get("SA3_MODEL", "stabilityai/stable-audio-3-medium-base"), sr=sr, hop=hop, fps=fps,
                clip_s=n_samp / sr, n_files={k: len(v) for k, v in split.items()}, files={k: [w for w, _ in v] for k, v in split.items()})
    t0 = time.time()
    for name, plist in split.items():
        samples = []
        for wav, midi in plist:
            info = torchaudio.info(wav); dur = info.num_frames / info.sample_rate
            starts = np.arange(20.0, max(20.0, dur - CLIP_S - 5.0), CLIP_S * 1.5)[:CLIPS_PER_FILE]   # spread over the piece
            for st in starts:
                y, sr0 = torchaudio.load(wav, frame_offset=int(st * info.sample_rate), num_frames=int(CLIP_S * info.sample_rate) + 4096)
                if sr0 != sr:
                    y = torchaudio.functional.resample(y, sr0, sr)
                y = y[:, :n_samp]
                if y.shape[-1] < n_samp:
                    continue
                if y.shape[0] == 1:
                    y = y.repeat(2, 1)
                with torch.no_grad():
                    z = model.pretransform.encode(y[None].to("cuda").to(next(model.parameters()).dtype))   # (1, C, T)
                z = z.float().cpu()
                T = z.shape[-1]
                lab = midi_to_pitch_class(midi, float(st), n_samp / sr, T / (n_samp / sr))        # (T', 12) at the latent fps
                if lab.shape[0] != T:
                    lab = torch.nn.functional.interpolate(torch.from_numpy(lab).T[None], size=T, mode="nearest")[0].T.numpy()
                samples.append({"audio_latent": z, "pitch_label": torch.from_numpy(lab.astype(np.float32)), "file": wav, "start": float(st), "fps": T / (n_samp / sr)})
            print(f"  {name}: {len(samples)} clips  [{time.time()-t0:.0f}s]  last latent {tuple(z.shape)}", flush=True)
        torch.save(samples, OUT / f"{name}.pt")
        meta[f"n_{name}"] = len(samples)
        allz = torch.cat([s["audio_latent"] for s in samples], dim=-1)
        meta[f"{name}_latent_stats"] = dict(mean=float(allz.mean()), std=float(allz.std()), rms=float(allz.pow(2).mean().sqrt()),
                                           T=int(samples[0]["audio_latent"].shape[-1]), C=int(samples[0]["audio_latent"].shape[1]))
        meta[f"{name}_label_density"] = float(torch.stack([s["pitch_label"] for s in samples]).mean())
    json.dump(meta, open(RES / "dataset.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "files"}, indent=1))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
