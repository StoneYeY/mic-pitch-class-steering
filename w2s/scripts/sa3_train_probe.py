"""Train the pitch-class CNN probe on Stable Audio 3 latents (MLSP recipe, RF-parameterised noise augmentation).

MLSP trained on z + sigma * eps (sigma in {0, .1, .3, .5, .7}, RMS re-normalised).  A rectified-flow sampler sees
x_t = (1 - t) z + t eps, whose noise/signal amplitude ratio is t/(1-t); the equivalent levels are
t in {0, .09, .23, .33, .41}.  We train on those (SA3_T_LEVELS to override) and evaluate micro-F1 per level.
Writes checkpoints/sa3_probe_best.pt (+ arch), results/<job>/train.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from w2s.sa3 import make_probe  # noqa: E402

RES = Path(os.environ.get("W2S_RESULTS", "results/sa3_probe")); RES.mkdir(parents=True, exist_ok=True)
DATA = Path(os.environ.get("SA3_DATA", "data/sa3_probe"))
CKPT = Path(os.environ.get("SA3_PROBE", "checkpoints/sa3_probe_best.pt")); CKPT.parent.mkdir(parents=True, exist_ok=True)
EPOCHS = int(os.environ.get("SA3_EPOCHS", "50")); BS = int(os.environ.get("SA3_BS", "16"))
LR = float(os.environ.get("SA3_LR", "1e-3")); WD = float(os.environ.get("SA3_WD", "1e-5"))
HIDDEN = int(os.environ.get("SA3_HIDDEN", "128"))
T_LEVELS = [float(x) for x in os.environ.get("SA3_T_LEVELS", "0,0.09,0.23,0.33,0.41").split(",")]
EVAL_T = [0.0, 0.09, 0.23, 0.33, 0.41, 0.5, 0.6, 0.7, 0.8]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load(name):
    s = torch.load(DATA / f"{name}.pt", weights_only=False)
    Z = torch.cat([x["audio_latent"] for x in s], 0)                 # (N, C, T)
    Y = torch.stack([x["pitch_label"] for x in s], 0)                # (N, T, 12)
    return Z, Y


def interp(z, t):
    """rectified-flow interpolant with a fresh Gaussian; t=0 -> clean."""
    if t <= 0:
        return z
    return (1 - t) * z + t * torch.randn_like(z)


@torch.no_grad()
def f1_at(probe, Z, Y, t, thr=0.5, bs=64):
    probe.eval(); tp = fp = fn = 0.0; bce = 0.0; n = 0
    g = torch.Generator(device=dev).manual_seed(0)
    for i in range(0, len(Z), bs):
        z = Z[i:i + bs].to(dev); y = Y[i:i + bs].to(dev)
        if t > 0:
            z = (1 - t) * z + t * torch.randn(z.shape, generator=g, device=dev)
        logits = probe(z); p = torch.sigmoid(logits)
        bce += nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum").item(); n += y.numel()
        pred = p >= thr
        tp += (pred & (y > 0)).sum().item(); fp += (pred & (y == 0)).sum().item(); fn += (~pred & (y > 0)).sum().item()
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    return dict(f1=2 * prec * rec / (prec + rec + 1e-9), precision=prec, recall=rec, bce=bce / n)


def main():
    Ztr, Ytr = load("train"); Zva, Yva = load("val")
    C = Ztr.shape[1]
    print(f"train {tuple(Ztr.shape)} val {tuple(Zva.shape)}  label density {Ytr.mean():.3f}")
    probe = make_probe(in_channels=C, hidden=HIDDEN).to(dev)
    n_params = sum(p.numel() for p in probe.parameters())
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.BCEWithLogitsLoss()
    hist = []; best = float("inf"); t0 = time.time()
    rng = np.random.default_rng(0)
    for ep in range(EPOCHS):
        probe.train(); perm = torch.randperm(len(Ztr)); tot = 0.0; nb = 0
        for i in range(0, len(Ztr), BS):
            idx = perm[i:i + BS]
            z = Ztr[idx].to(dev); y = Ytr[idx].to(dev)
            z = interp(z, float(rng.choice(T_LEVELS)))
            loss = crit(probe(z), y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(probe.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        va = {f"t{t}": f1_at(probe, Zva, Yva, t) for t in (0.0, 0.23, 0.41)}
        vloss = float(np.mean([v["bce"] for v in va.values()]))
        hist.append(dict(epoch=ep + 1, train_loss=tot / nb, val_bce=vloss, **{k + "_f1": v["f1"] for k, v in va.items()}))
        if vloss < best:
            best = vloss
            torch.save({"state_dict": probe.state_dict(), "arch": dict(in_channels=C, hidden=HIDDEN, kernel_size=5, num_layers=2),
                        "epoch": ep + 1, "val_bce": vloss, "t_levels": T_LEVELS, "n_params": n_params}, CKPT)
        print(f"ep {ep+1:3d} train {tot/nb:.4f} val_bce {vloss:.4f} f1@t0 {va['t0.0']['f1']:.3f} f1@t.23 {va['t0.23']['f1']:.3f} "
              f"f1@t.41 {va['t0.41']['f1']:.3f} [{time.time()-t0:.0f}s]", flush=True)
    ck = torch.load(CKPT, weights_only=False); probe.load_state_dict(ck["state_dict"])
    final = {f"t{t}": f1_at(probe, Zva, Yva, t) for t in EVAL_T}
    out = dict(n_params=n_params, in_channels=C, hidden=HIDDEN, best_epoch=ck["epoch"], t_levels=T_LEVELS, val=final, history=hist,
               n_train=len(Ztr), n_val=len(Zva))
    json.dump(out, open(RES / "train.json", "w"), indent=1)
    print(json.dumps({k: (round(v["f1"], 3) if isinstance(v, dict) and "f1" in v else v) for k, v in final.items()}))
    print(f"probe {n_params/1e3:.0f}k params, best epoch {ck['epoch']}, saved {CKPT}")


if __name__ == "__main__":
    main()
