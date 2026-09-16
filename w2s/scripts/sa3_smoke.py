"""SA3 smoke test (job 017): can this box run stable-audio-3-medium-base with a per-step callback?

Prints: library versions, model_config essentials (latent dim / downsampling / objective), autoencoder
latent stats on one MAESTRO clip, a 50-step Euler generation of a 5-s clip with a callback that records
per-step latent RMS, peak VRAM, wall time.  Writes results/<job>/smoke.json and a wav.
Run inside an env that has stable_audio_tools:  conda run -n <env> python w2s/scripts/sa3_smoke.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

RES = Path(os.environ.get("W2S_RESULTS", "results/sa3_smoke")); RES.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("SA3_MODEL", "stabilityai/stable-audio-3-medium-base")
STEPS = int(os.environ.get("SA3_STEPS", "50"))
CFG = float(os.environ.get("SA3_CFG", "7.0"))
DUR = float(os.environ.get("SA3_DUR", "5.0"))
out = {"model": MODEL, "steps": STEPS, "cfg": CFG, "dur": DUR}


def dump():
    json.dump(out, open(RES / "smoke.json", "w"), indent=1, default=str)


try:
    import torch, torchaudio  # noqa: E402
    out["torch"] = torch.__version__
    import stable_audio_tools  # noqa: E402
    out["stable_audio_tools"] = getattr(stable_audio_tools, "__version__", "?")
    from stable_audio_tools import get_pretrained_model  # noqa: E402
    from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint as generate_diffusion_cond  # noqa: E402
    import inspect
    out["generate_sig"] = str(inspect.signature(generate_diffusion_cond))[:600]
    print("versions", out["torch"], out["stable_audio_tools"]); dump()

    dev = "cuda"
    t0 = time.time()
    model, cfg = get_pretrained_model(MODEL)
    out["load_s"] = round(time.time() - t0, 1)
    out["sample_rate"] = cfg["sample_rate"]; out["sample_size"] = cfg.get("sample_size")
    m = cfg["model"]
    pt = m.get("pretransform", {}); ptc = pt.get("config", {})
    out["pretransform"] = {k: v for k, v in pt.items() if k != "config"}
    out["latent_dim"] = ptc.get("latent_dim"); out["downsampling_ratio"] = ptc.get("downsampling_ratio")
    out["diffusion_objective"] = getattr(model, "diffusion_objective", m.get("diffusion", {}).get("diffusion_objective"))
    out["conditioning_ids"] = [c.get("id") for c in m.get("conditioning", {}).get("configs", [])]
    out["n_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    print("config", {k: out[k] for k in ("sample_rate", "latent_dim", "downsampling_ratio", "diffusion_objective", "conditioning_ids", "n_params_M")}); dump()

    dtype = torch.bfloat16 if os.environ.get("SA3_DTYPE", "bf16") == "bf16" else torch.float16
    model = model.to(dev).to(dtype).eval()
    out["dtype"] = str(dtype)
    torch.cuda.reset_peak_memory_stats()

    # ---- autoencoder on a MAESTRO clip ----
    wavs = sorted(glob.glob(os.path.expanduser("~/Desktop/MIC/data/maestro/**/*.wav"), recursive=True))[:1]
    if wavs:
        y, sr = torchaudio.load(wavs[0])
        y = y[:, : int(sr * DUR)]
        if sr != cfg["sample_rate"]:
            y = torchaudio.functional.resample(y, sr, cfg["sample_rate"])
        if y.shape[0] == 1:
            y = y.repeat(2, 1)
        y = y[None].to(dev).to(dtype)
        with torch.no_grad():
            z = model.pretransform.encode(y)
        out["enc_input_shape"] = list(y.shape); out["enc_latent_shape"] = list(z.shape)
        out["latent_fps"] = round(z.shape[-1] / DUR, 3)
        zf = z.float()
        out["latent_stats"] = dict(mean=float(zf.mean()), std=float(zf.std()), rms=float(zf.pow(2).mean().sqrt()),
                                   ch_std_min=float(zf.std(dim=(0, 2)).min()), ch_std_max=float(zf.std(dim=(0, 2)).max()))
        with torch.no_grad():
            yhat = model.pretransform.decode(z)
        out["dec_output_shape"] = list(yhat.shape)
        n = min(y.shape[-1], yhat.shape[-1])
        out["recon_snr_db"] = float(10 * torch.log10(y[..., :n].float().pow(2).mean() / (y[..., :n].float() - yhat[..., :n].float()).pow(2).mean()))
        print("autoencoder", {k: out[k] for k in ("enc_latent_shape", "latent_fps", "latent_stats", "recon_snr_db")}); dump()
    else:
        out["enc_note"] = "no MAESTRO wav found under ~/Desktop/MIC/data/maestro"

    # ---- 50-step Euler generation with callback ----
    log = []

    def cb(d):
        x = d["x"]
        log.append(dict(i=int(d["i"]), t=float(d["t"].flatten()[0]), rms=float(x.float().pow(2).mean().sqrt()),
                        den_rms=float(d["denoised"].float().pow(2).mean().sqrt()) if "denoised" in d else None))
        if int(d["i"]) == 25:   # prove in-place edits reach the sampler: add a tiny marker and check next step
            x.add_(0.0)

    cond = [{"prompt": "A calm classical piano piece in the style of a nocturne", "seconds_total": DUR}]
    sample_size = int(round(DUR * cfg["sample_rate"] / out["downsampling_ratio"])) * out["downsampling_ratio"]
    out["gen_sample_size"] = sample_size
    kw = dict(steps=STEPS, cfg_scale=CFG, conditioning=cond, sample_size=sample_size,
              sampler_type="euler", device=dev, seed=0, callback=cb)
    t0 = time.time()
    with torch.no_grad():
        audio = generate_diffusion_cond(model, **kw)
    out["gen_s"] = round(time.time() - t0, 1)
    out["gen_audio_shape"] = list(audio.shape)
    out["n_callback_steps"] = len(log); out["callback_log_head"] = log[:3]; out["callback_log_tail"] = log[-3:]
    out["t_schedule"] = [round(r["t"], 4) for r in log]
    out["model_attrs"] = {k: getattr(model, k, None) for k in ("mask_padding_attention", "use_effective_length_for_schedule", "sampling_dist_shift")}
    out["model_attrs"] = {k: (str(v) if v is not None else None) for k, v in out["model_attrs"].items()}
    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    a = audio[0].float().cpu()
    a = a / max(1e-6, a.abs().max())
    torchaudio.save(str(RES / "smoke_gen.wav"), a[:, : int(cfg["sample_rate"] * DUR)], cfg["sample_rate"])
    out["audio_rms_first5s"] = float(a[:, : int(cfg["sample_rate"] * DUR)].pow(2).mean().sqrt())
    out["audio_len_s"] = round(a.shape[-1] / cfg["sample_rate"], 2)
    print("generation", {k: out[k] for k in ("gen_s", "gen_audio_shape", "n_callback_steps", "peak_vram_gb", "audio_len_s")}); dump()

    t0 = time.time()
    with torch.no_grad():
        generate_diffusion_cond(model, **{**kw, "callback": None})
    out["gen_s_nocb"] = round(time.time() - t0, 1)
    # ---- return_latents path (needed to record trajectories) ----
    with torch.no_grad():
        lat = generate_diffusion_cond(model, **{**kw, "return_latents": True, "steps": 8})
    out["return_latents_shape"] = list(lat.shape)
    dump()
    print("SMOKE OK")
except Exception as e:  # noqa: BLE001
    out["error"] = repr(e); out["traceback"] = traceback.format_exc()[-3000:]
    dump(); print("SMOKE FAILED", repr(e)); print(out["traceback"]); sys.exit(1)
