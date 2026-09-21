"""Stable Audio 3 backend (stable-audio-tools, rectified-flow DiT) with the same interface as
w2s.generate.W2SGenerator, so that the Exp A/B/C/D scripts run unchanged via w2s.backend.make_generator().

Differences from the SAO 1.0 backend that matter for the analysis
  * latent: (256, T) at sample_rate/4096 = 10.77 frames/s (SAO: (64, T) at 21.5 frames/s); the latent
    covers exactly the requested clip, so every frame is "audible" (n_aud == T).
  * sampler: discrete Euler on the rectified-flow schedule t = 1 -> 0 in N steps (sigma_max = 1).
    stable_audio_tools calls the callback AFTER the velocity v(x_i, t_i) is computed and BEFORE the
    Euler update x_{i+1} = x_i + dt v.  We probe x_i (noise level t_i) and guide it in place, so an update
    at callback i is injected between model calls i and i+1 -- exactly where the diffusers callback of
    the SAO backend injects it.  The stored trajectory row i is the post-update latent x_{i+1} (as for
    SAO): it is the `x` seen by callback i+1 before any guidance, and row N-1 is the final latent.
  * entry point: generate_diffusion_cond_inpaint (the medium-base checkpoint is a cond_inpaint model and
    expects an all-zero inpaint mask for plain generation); the timestep schedule may be warped by the
    model's sampling_dist_shift, so nothing here assumes dt = -1/N.
  * guidance rule, reliability, schedules, targets: identical code paths (Eq. 1 / Eq. 2 of the paper).
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import Melody
from .generate import RunResult
from .reliability import reliability
from .schedules import GuidanceSchedule

SA3_MODEL = os.environ.get("SA3_MODEL", "stabilityai/stable-audio-3-medium-base")
SA3_PROBE = os.environ.get("SA3_PROBE", "checkpoints/sa3_probe_best.pt")
SA3_CFG = float(os.environ.get("SA3_CFG", "7.0"))
SA3_STEPS = int(os.environ.get("SA3_STEPS", "50"))
DTYPE = torch.bfloat16 if os.environ.get("SA3_DTYPE", "bf16") == "bf16" else torch.float16


def load_sa3(device: str = "cuda"):
    from stable_audio_tools import get_pretrained_model
    model, cfg = get_pretrained_model(SA3_MODEL)
    model = model.to(device).to(DTYPE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def make_probe(in_channels: int = 256, hidden: int = 128, kernel_size: int = 5, num_layers: int = 2):
    from src.probe import CNNProbe   # the MLSP architecture, wider input
    return CNNProbe(in_channels=in_channels, hidden_channels=hidden, out_classes=12, kernel_size=kernel_size, num_layers=num_layers)


def load_sa3_probe(path: str | Path | None = None, device: str = "cuda") -> torch.nn.Module:
    path = Path(path or SA3_PROBE)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model_state_dict", ck)) if isinstance(ck, dict) else ck
    arch = ck.get("arch", {}) if isinstance(ck, dict) else {}
    probe = make_probe(**{k: arch[k] for k in ("in_channels", "hidden", "kernel_size", "num_layers") if k in arch})
    probe.load_state_dict(sd)
    return probe.to(device).eval()


class SA3Generator:
    def __init__(self, model, cfg: dict, probe: torch.nn.Module, device: str | torch.device | None = None,
                 alpha: float = 0.05, reliability_mode: str = "entropy", audio_length: float = 5.0):
        self.model, self.cfg = model, cfg
        self.probe = probe.eval()
        self.device = torch.device(device) if device is not None else next(probe.parameters()).device
        self.alpha = alpha
        self.reliability_mode = reliability_mode
        self.random_grad = False
        self.sr = int(cfg["sample_rate"])
        self.hop = int(cfg["model"]["pretransform"]["config"]["downsampling_ratio"])   # 4096
        self.latent_dim = int(cfg["model"]["pretransform"]["config"]["latent_dim"])   # 256
        self.fps = self.sr / self.hop                                                  # 10.77 Hz
        self.T = int(math.ceil(audio_length * self.sr / self.hop))                    # 54 frames for 5 s
        self.sample_size = self.T * self.hop
        self.audio_length = audio_length

    # ------------------------------------------------------------------ targets
    def make_target(self, melody: Melody | None, audio_length: float, target_fps: float | None = None):
        fps = self.fps if target_fps is None else float(target_fps)
        n_aud = min(self.T, int(math.ceil(audio_length * self.fps)))
        tgt = torch.zeros(1, self.T, 12, device=self.device)
        if melody is not None:
            beat = 60.0 / melody.tempo * melody.note_beats
            for i, n in enumerate(melody.notes):
                t0, t1 = i * beat, (i + 1) * beat
                if t0 >= audio_length:
                    break
                f0, f1 = int(t0 * fps), int(t1 * fps)
                f1 = min(max(f1, f0 + 1), self.T)
                tgt[0, f0:f1, n % 12] = 1.0
        return tgt, n_aud

    # ------------------------------------------------------------------ one generation
    def run(self, prompt: str, seed: int, melody: Melody | None, schedule: GuidanceSchedule | None, *,
            steps: int = SA3_STEPS, cfg: float = SA3_CFG, audio_length: float | None = None, negative_prompt: str = "",
            loss_region: str = "audible", save_trajectory: bool = False, verbose: bool = False,
            target_fps: float | None = None) -> RunResult:
        from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint
        dev = self.device
        audio_length = audio_length or self.audio_length
        target, n_aud = self.make_target(melody, audio_length, target_fps)
        region = slice(0, n_aud) if loss_region == "audible" else slice(0, self.T)
        frame_mask = torch.zeros(self.T, dtype=torch.bool, device=dev); frame_mask[:n_aud] = True
        guide_on = schedule is not None and not schedule.is_null and melody is not None
        traj = torch.empty((steps, self.latent_dim, self.T), dtype=torch.float16) if save_trajectory else None
        log: list[dict] = []
        ts: list[float] = []
        state = {"n_used": 0}

        def callback(d: dict):
            x = d["x"]                                   # (1, C, T) model dtype, PRE-update latent x_i
            i = int(d["i"]); t_i = float(d["t"].flatten()[0])
            z32 = x.detach().float()
            with torch.no_grad():
                p = torch.sigmoid(self.probe(z32)).permute(0, 2, 1)          # (1,12,T)
                R = float(reliability(p, self.reliability_mode, frame_mask)[0])
            rec = {"step": i, "R": R, "rms_z": float(z32.pow(2).mean().sqrt()), "sigma": t_i,
                   "guided": 0, "lam": 0.0, "loss": float("nan"), "upd_rms": 0.0}
            if traj is not None and i >= 1:
                traj[i - 1] = z32[0].to(torch.float16).cpu()       # x_i = post-update latent of step i-1
            ts.append(t_i)
            if guide_on and schedule.should_guide(i, R, state["n_used"]):
                lam = schedule.strength(i, R)
                with torch.enable_grad():
                    zg = z32.detach().clone().requires_grad_(True)
                    logits = self.probe(zg)                                     # (1,T,12)
                    loss = F.binary_cross_entropy_with_logits(logits[:, region], target[:, region])
                    grad = torch.autograd.grad(loss, zg)[0]
                g_reg, z_reg = grad[..., region], z32[..., region]
                g_rms = g_reg.pow(2).mean().sqrt(); z_rms = z_reg.pow(2).mean().sqrt()
                update = lam * z_rms * grad / (g_rms + 1e-8)
                update = torch.clamp(update, -self.alpha * z_rms, self.alpha * z_rms)
                if self.random_grad:   # control: a random direction with exactly the norm of the probe update
                    rnd = torch.randn_like(update)
                    update = rnd * (update.norm() / (rnd.norm() + 1e-8))
                with torch.no_grad():
                    x.sub_(update.to(x.dtype))                                   # in place -> reaches the sampler
                state["n_used"] += 1
                rec.update(guided=1, lam=float(lam), loss=float(loss.detach()),
                           upd_rms=float(update[..., region].pow(2).mean().sqrt()))
                if verbose:
                    print(f"  step {i:2d} t={t_i:.3f} R={R:.3f} lam={lam:.3f} loss={float(loss):.4f}")
            log.append(rec)

        cond = [{"prompt": prompt, "seconds_total": float(audio_length)}]
        neg = [{"prompt": negative_prompt, "seconds_total": float(audio_length)}] if negative_prompt else None
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev)
        t0 = time.time()
        with torch.no_grad():
            lat = generate_diffusion_cond_inpaint(self.model, steps=steps, cfg_scale=cfg, conditioning=cond,
                                                  negative_conditioning=neg, sample_size=self.sample_size,
                                                  sampler_type="euler", device=str(dev), seed=int(seed), callback=callback,
                                                  return_latents=True, disable_tqdm=True)
            lat = lat.detach()
            audio = self.model.pretransform.decode(lat.to(next(self.model.parameters()).dtype))   # (1, 2, samples)
        if traj is not None:
            traj[steps - 1] = lat[0].detach().float().to(torch.float16).cpu()
            # make the per-step log describe the stored (post-update) latents, as in the SAO backend
            for i in range(steps):
                log[i]["sigma"] = ts[i + 1] if i + 1 < len(ts) else 0.0
                log[i]["rms_z"] = float(traj[i].float().pow(2).mean().sqrt())
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        runtime = time.time() - t0
        n_samp = int(round(audio_length * self.sr))
        a = audio[0].float().cpu().numpy().T[:n_samp]                              # (samples, channels)
        peak = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0
        return RunResult(audio=a.astype(np.float32), sr=self.sr, n_updates=state["n_used"], runtime_s=runtime,
                         peak_vram_gb=peak, log=log, trajectory=traj.numpy() if traj is not None else None,
                         final_latent=lat[0].detach().to(torch.float16).cpu().numpy(), target=target[0].T.cpu().numpy(),
                         n_audible=n_aud,
                         meta={"prompt": prompt, "seed": seed, "melody": melody.name if melody else None,
                               "schedule": schedule.to_dict() if schedule else None, "steps": steps, "cfg": cfg,
                               "audio_length": audio_length, "loss_region": loss_region, "alpha": self.alpha,
                               "fps": self.fps, "T": self.T, "target_fps": target_fps, "backend": "sa3",
                               "model": SA3_MODEL, "sampler": "euler"})

    # ------------------------------------------------------------------ helpers
    def probe_fn(self, z: torch.Tensor) -> torch.Tensor:
        """(1,C,T) -> (1,12,T) probabilities (for w2s.probe_eval)."""
        with torch.no_grad():
            return torch.sigmoid(self.probe(z.to(self.device).float())).permute(0, 2, 1)
