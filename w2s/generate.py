"""Schedule-driven probe guidance for Stable Audio Open (diffusers StableAudioPipeline).

Fixes the target/latent time-axis alignment of the MLSP implementation: the pipeline's
latent is ALWAYS transformer.sample_size (=1024) frames at sampling_rate/hop (=21.53 Hz),
i.e. 47.55 s, of which only the first `audio_length` seconds are decoded.  The target
therefore lives on the first ceil(audio_length * fps) frames and the loss is computed
on that "audible" region (loss_region="audible"), or on all frames (loss_region="all",
the MLSP behaviour but with the correct fps).

The update rule is MLSP's (RMS-normalised gradient step, element-wise clamp at
alpha * rms(z)); only WHEN it fires and HOW STRONG (lambda_t) come from a
w2s.schedules.GuidanceSchedule.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .data import Melody
from .reliability import reliability
from .schedules import GuidanceSchedule


@dataclass
class RunResult:
    audio: np.ndarray                 # (samples, channels) float32
    sr: int
    n_updates: int
    runtime_s: float
    peak_vram_gb: float
    log: list[dict] = field(default_factory=list)      # one entry per sampler step
    trajectory: np.ndarray | None = None               # (N, 64, T) float16, post-update latents
    final_latent: np.ndarray | None = None             # (64, T) float16, latent decoded to audio
    target: np.ndarray | None = None                   # (12, T_latent) float32
    n_audible: int = 0
    meta: dict = field(default_factory=dict)


class W2SGenerator:
    def __init__(self, pipe, probe: torch.nn.Module, device: str | torch.device | None = None,
                 alpha: float = 0.05, reliability_mode: str = "entropy"):
        self.pipe = pipe
        self.probe = probe.eval()
        self.device = torch.device(device) if device is not None else next(probe.parameters()).device
        self.alpha = alpha                       # MAX_UPDATE_RATIO in MLSP
        self.reliability_mode = reliability_mode
        self.sr = int(pipe.vae.config.sampling_rate)
        self.hop = int(pipe.vae.hop_length)      # 2048 for SAO
        self.fps = self.sr / self.hop            # 21.53 Hz
        self.T = int(pipe.transformer.config.sample_size)   # 1024

    # ------------------------------------------------------------------ targets
    def make_target(self, melody: Melody | None, audio_length: float,
                    target_fps: float | None = None) -> tuple[torch.Tensor, int]:
        """(1, T, 12) target on the latent grid and the number of audible frames.
        target_fps=None -> the correct latent frame rate (sr/hop = 21.53 Hz).
        target_fps=T/audio_length (=204.8) reproduces the MLSP code's placement (legacy bug)."""
        fps = self.fps if target_fps is None else float(target_fps)
        n_aud = int(math.ceil(audio_length * self.fps))
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
            steps: int = 50, cfg: float = 4.0, audio_length: float = 5.0, negative_prompt: str = "",
            loss_region: str = "audible", save_trajectory: bool = False, verbose: bool = False,
            target_fps: float | None = None) -> RunResult:
        dev = self.device
        target, n_aud = self.make_target(melody, audio_length, target_fps)
        region = slice(0, n_aud) if loss_region == "audible" else slice(0, self.T)
        frame_mask = torch.zeros(self.T, dtype=torch.bool, device=dev)
        frame_mask[:n_aud] = True
        guide_on = schedule is not None and not schedule.is_null and melody is not None
        traj = torch.empty((steps, self.pipe.transformer.config.in_channels, self.T), dtype=torch.float16) \
            if save_trajectory else None
        log: list[dict] = []
        state = {"n_used": 0}
        sigmas = getattr(self.pipe.scheduler, "sigmas", None)

        def callback(step_idx: int, timestep, latents: torch.Tensor):
            z32 = latents.float()
            with torch.no_grad():
                p = torch.sigmoid(self.probe(z32)).permute(0, 2, 1)          # (1,12,T)
                R = float(reliability(p, self.reliability_mode, frame_mask)[0])
            rec = {"step": step_idx, "R": R, "rms_z": float(z32.pow(2).mean().sqrt()),
                   "sigma": float(sigmas[min(step_idx + 1, len(sigmas) - 1)]) if sigmas is not None else float("nan"),
                   "guided": 0, "lam": 0.0, "loss": float("nan"), "upd_rms": 0.0}
            if guide_on and schedule.should_guide(step_idx, R, state["n_used"]):
                lam = schedule.strength(step_idx, R)
                with torch.enable_grad():
                    zg = z32.detach().clone().requires_grad_(True)
                    logits = self.probe(zg)                                     # (1,T,12)
                    loss = F.binary_cross_entropy_with_logits(logits[:, region], target[:, region])
                    grad = torch.autograd.grad(loss, zg)[0]
                g_reg, z_reg = grad[..., region], z32[..., region]
                g_rms = g_reg.pow(2).mean().sqrt()
                z_rms = z_reg.pow(2).mean().sqrt()
                update = lam * z_rms * grad / (g_rms + 1e-8)
                update = torch.clamp(update, -self.alpha * z_rms, self.alpha * z_rms)
                with torch.no_grad():
                    latents.sub_(update.to(latents.dtype))
                state["n_used"] += 1
                rec.update(guided=1, lam=float(lam), loss=float(loss.detach()),
                           upd_rms=float(update[..., region].pow(2).mean().sqrt()))
                if verbose:
                    print(f"  step {step_idx:2d} R={R:.3f} lam={lam:.3f} loss={float(loss):.4f}")
            if traj is not None:
                traj[step_idx] = latents.detach()[0].to(torch.float16).cpu()
            if step_idx == steps - 1:
                state["final"] = latents.detach()[0].to(torch.float16).cpu().numpy()
            log.append(rec)

        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize(dev)
        t0 = time.time()
        gen = torch.Generator(dev).manual_seed(int(seed))
        out = self.pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
                        guidance_scale=cfg, audio_end_in_s=audio_length, generator=gen,
                        callback=callback, callback_steps=1)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        runtime = time.time() - t0
        audio = out.audios[0].T.float().cpu().numpy()
        peak = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0
        return RunResult(audio=audio, sr=self.sr, n_updates=state["n_used"], runtime_s=runtime,
                         peak_vram_gb=peak, log=log, trajectory=traj.numpy() if traj is not None else None,
                         final_latent=state.get("final"), target=target[0].T.cpu().numpy(), n_audible=n_aud,
                         meta={"prompt": prompt, "seed": seed, "melody": melody.name if melody else None,
                               "schedule": schedule.to_dict() if schedule else None, "steps": steps,
                               "cfg": cfg, "audio_length": audio_length, "loss_region": loss_region,
                               "alpha": self.alpha, "fps": self.fps, "T": self.T,
                               "target_fps": target_fps})

    # ------------------------------------------------------------------ helpers
    def probe_fn(self, z: torch.Tensor) -> torch.Tensor:
        """(1,64,T) -> (1,12,T) probabilities (for w2s.probe_eval)."""
        with torch.no_grad():
            return torch.sigmoid(self.probe(z.to(self.device).float())).permute(0, 2, 1)
