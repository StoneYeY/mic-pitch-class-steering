"""
Probe-Guided Inference for MIC

Contribution 2: Use trained probes as learned loss functions
for inference-time latent steering.

This module integrates probe guidance into the diffusion
denoising process without retraining the base model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Callable, Union, Dict
from pathlib import Path
import math

from .utils import notes_to_pitch_class_target, compute_rms


class ProbeGuidedGenerator:
    """
    Probe-guided music generation.

    Uses a trained probe as a learned loss function to steer
    the diffusion denoising process toward target pitch content.
    """

    def __init__(
        self,
        pipe,  # StableAudioPipeline
        probe: nn.Module,
        device: str = "cuda",
    ):
        """
        Args:
            pipe: Loaded StableAudioPipeline
            probe: Trained probe network
            device: Device
        """
        self.pipe = pipe
        self.probe = probe.to(device).eval()
        self.device = device

        # Infer latent parameters from pipeline
        self.sample_rate = getattr(pipe.vae.config, 'sampling_rate', 44100)

    def generate(
        self,
        prompt: str,
        target_notes: List[int],
        tempo: float = 120.0,
        note_duration_beats: float = 1.0,
        audio_length: float = 5.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        probe_guidance_scale: float = 0.05,
        probe_guidance_start: float = 0.7,
        probe_guidance_end: float = 1.0,
        probe_guidance_every_n: int = 2,
        max_update_ratio: float = 0.05,
        seed: Optional[int] = 42,
        negative_prompt: str = "",
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Generate audio with probe guidance.

        Args:
            prompt: Text prompt
            target_notes: List of MIDI pitch numbers
            tempo: BPM
            note_duration_beats: Duration of each note in beats
            audio_length: Output audio length in seconds
            num_inference_steps: Diffusion steps
            guidance_scale: CFG scale
            probe_guidance_scale: Probe loss gradient scale
            probe_guidance_start: Start probe guidance at this fraction of denoising
            probe_guidance_end: End probe guidance at this fraction
            probe_guidance_every_n: Apply guidance every N steps
            max_update_ratio: Max update as fraction of latent RMS
            seed: Random seed
            negative_prompt: Negative prompt
            verbose: Print progress

        Returns:
            Generated audio as numpy array (samples, channels)
        """
        # Create callback with probe guidance
        callback = self._make_probe_callback(
            target_notes=target_notes,
            tempo=tempo,
            note_duration_beats=note_duration_beats,
            audio_length=audio_length,
            num_inference_steps=num_inference_steps,
            probe_guidance_scale=probe_guidance_scale,
            probe_guidance_start=probe_guidance_start,
            probe_guidance_end=probe_guidance_end,
            probe_guidance_every_n=probe_guidance_every_n,
            max_update_ratio=max_update_ratio,
            verbose=verbose,
        )

        # Generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        # Run inference
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            audio_end_in_s=audio_length,
            callback=callback,
            callback_steps=1,
            generator=generator,
        )

        # Convert to numpy
        audio = output.audios[0].T.float().cpu().numpy()

        return audio

    def _make_probe_callback(
        self,
        target_notes: List[int],
        tempo: float,
        note_duration_beats: float,
        audio_length: float,
        num_inference_steps: int,
        probe_guidance_scale: float,
        probe_guidance_start: float,
        probe_guidance_end: float,
        probe_guidance_every_n: int,
        max_update_ratio: float,
        verbose: bool,
    ) -> Callable:
        """
        Create callback function for probe-guided denoising.

        The callback computes probe loss and applies gradient
        descent to steer the latent toward target pitch content.
        """
        # Compute step range
        start_step = int(probe_guidance_start * num_inference_steps)
        end_step = int(probe_guidance_end * num_inference_steps)
        if start_step >= num_inference_steps:
            raise ValueError(
                f"probe_guidance_start={probe_guidance_start} leaves no steps for guidance "
                f"(start_step={start_step} >= num_inference_steps={num_inference_steps})"
            )
        if end_step < start_step:
            raise ValueError(
                f"probe_guidance_end={probe_guidance_end} is before probe_guidance_start={probe_guidance_start} "
                f"(end_step={end_step} < start_step={start_step})"
            )

        # Target will be computed lazily when we know latent shape
        target_pc = None
        latent_len = None

        def probe_callback(step_idx: int, timestep: Union[int, float], latents: torch.Tensor):
            nonlocal target_pc, latent_len

            # Skip if outside guidance range
            if step_idx < start_step or step_idx > end_step:
                return

            # Skip based on frequency
            if (step_idx - start_step) % probe_guidance_every_n != 0:
                return

            # Lazily compute target when we know latent shape
            if target_pc is None or latent_len != latents.shape[-1]:
                latent_len = latents.shape[-1]
                target_pc = notes_to_pitch_class_target(
                    notes=target_notes,
                    tempo=tempo,
                    note_duration_beats=note_duration_beats,
                    total_frames=latent_len,
                    audio_length=audio_length,
                    device=self.device,
                )

            # ========== Probe Guidance ==========
            # Pipeline runs under @torch.no_grad(); re-enable for gradient computation.
            # All intermediate math stays float32 to avoid fp16 overflow/underflow.
            latents_f32 = latents.float()
            with torch.enable_grad():
                latents_for_grad = latents_f32.detach().clone().requires_grad_(True)
                logits = self.probe(latents_for_grad)  # (B, T, n_classes)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    target_pc.expand(latents.shape[0], -1, -1),
                    reduction='mean'
                )
                grad = torch.autograd.grad(loss, latents_for_grad)[0]  # float32

            # ========== RMS-Normalized Gradient Update ==========
            grad_rms = compute_rms(grad)
            latent_rms = compute_rms(latents_f32)

            normalized_grad = grad / (grad_rms + 1e-8)
            update = probe_guidance_scale * latent_rms * normalized_grad
            max_update = max_update_ratio * latent_rms
            update = torch.clamp(update, -max_update, max_update)

            # Apply update in-place, converting back to original dtype
            with torch.no_grad():
                latents.sub_(update.to(latents.dtype))

            if verbose and (step_idx - start_step) % 10 == 0:
                print(f"  Step {step_idx}: probe_loss={loss.item():.4f}, "
                      f"update_rms={compute_rms(update).item():.4f}")

        return probe_callback


def make_combined_callback(
    pipe,
    probe: Optional[nn.Module] = None,
    target_notes: Optional[List[int]] = None,
    # Original ITC parameters
    use_latent_fusion: bool = True,
    fusion_notes: Optional[List[int]] = None,
    fusion_scale: float = 0.1,
    fusion_start: float = 0.5,
    fusion_end: float = 1.0,
    fusion_every_n: int = 5,
    # Probe guidance parameters
    use_probe_guidance: bool = True,
    probe_guidance_scale: float = 0.05,
    probe_guidance_start: float = 0.7,
    probe_guidance_end: float = 1.0,
    probe_guidance_every_n: int = 2,
    max_update_ratio: float = 0.05,
    # Common parameters
    tempo: float = 120.0,
    note_duration_beats: float = 1.0,
    audio_length: float = 5.0,
    num_inference_steps: int = 50,
    seed: int = 0,
) -> Callable:
    """
    Create combined callback with both latent fusion and probe guidance.

    This allows comparing and combining the two approaches:
    - Original MIC: variance-preserving latent fusion
    - New: probe-guided gradient steering

    Args:
        pipe: StableAudioPipeline
        probe: Trained probe (optional)
        target_notes: Target MIDI notes
        use_latent_fusion: Enable original latent fusion
        fusion_notes: Notes for fusion (uses target_notes if None)
        fusion_scale: Latent fusion strength
        fusion_start: Fusion start (fraction)
        fusion_end: Fusion end (fraction)
        use_probe_guidance: Enable probe guidance
        probe_guidance_scale: Probe gradient scale
        probe_guidance_start: Probe start (fraction)
        probe_guidance_end: Probe end (fraction)
        probe_guidance_every_n: Probe every N steps
        max_update_ratio: Max update ratio
        tempo: BPM
        note_duration_beats: Note duration in beats
        audio_length: Audio length in seconds
        num_inference_steps: Diffusion steps
        seed: User seed combined into fusion noise (step_idx + seed * 9973) for reproducibility

    Returns:
        Callback function
    """
    device = next(iter(pipe.transformer.parameters())).device
    dtype = next(iter(pipe.transformer.parameters())).dtype

    if fusion_notes is None:
        fusion_notes = target_notes

    # Compute step ranges with validation (O8)
    fusion_start_step = int(fusion_start * num_inference_steps)
    fusion_end_step = int(fusion_end * num_inference_steps)
    if fusion_end_step < fusion_start_step:
        raise ValueError(f"fusion_end={fusion_end} < fusion_start={fusion_start}")
    probe_start_step = int(probe_guidance_start * num_inference_steps)
    probe_end_step = int(probe_guidance_end * num_inference_steps)
    if use_probe_guidance and probe_start_step >= num_inference_steps:
        raise ValueError(
            f"probe_guidance_start={probe_guidance_start} leaves no steps for guidance "
            f"(probe_start_step={probe_start_step} >= num_inference_steps={num_inference_steps})"
        )
    if use_probe_guidance and probe_end_step < probe_start_step:
        raise ValueError(f"probe_guidance_end={probe_guidance_end} < probe_guidance_start={probe_guidance_start}")

    # Pre-compute control latent for fusion (if needed)
    control_latent = None
    if use_latent_fusion and fusion_notes:
        control_latent = _create_control_latent(
            pipe, fusion_notes, tempo, note_duration_beats, audio_length, device, dtype
        )

    # Target for probe (computed lazily)
    target_pc = None
    latent_len = None

    def combined_callback(step_idx: int, timestep: Union[int, float], latents: torch.Tensor):
        nonlocal target_pc, latent_len, control_latent

        # ========== Latent Fusion (Original MIC) ==========
        if use_latent_fusion and control_latent is not None:
            if fusion_start_step <= step_idx <= fusion_end_step:
                if (step_idx - fusion_start_step) % fusion_every_n == 0:
                    _apply_latent_fusion(
                        latents, control_latent, pipe, step_idx, fusion_scale, outer_seed=seed
                    )

        # ========== Probe Guidance (New) ==========
        if use_probe_guidance and probe is not None and target_notes:
            if probe_start_step <= step_idx <= probe_end_step:
                if (step_idx - probe_start_step) % probe_guidance_every_n == 0:
                    # Lazy compute target
                    if target_pc is None or latent_len != latents.shape[-1]:
                        latent_len = latents.shape[-1]
                        target_pc = notes_to_pitch_class_target(
                            notes=target_notes,
                            tempo=tempo,
                            note_duration_beats=note_duration_beats,
                            total_frames=latent_len,
                            audio_length=audio_length,
                            device=device,
                        )

                    _apply_probe_guidance(
                        latents, probe, target_pc, probe_guidance_scale, max_update_ratio
                    )

    return combined_callback


def _create_control_latent(pipe, notes, tempo, note_duration, audio_length, device, dtype):
    """Create control latent from notes via sonification + VAE encoding."""

    sample_rate = getattr(pipe.vae.config, 'sampling_rate', 44100)
    audio_samples = int(audio_length * sample_rate)

    # Sonify notes to sine waves
    audio = np.zeros(audio_samples, dtype=np.float32)
    sec_per_beat = 60.0 / tempo
    note_sec = note_duration * sec_per_beat

    t = 0.0
    for note in notes:
        if t >= audio_length:
            break
        freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
        start = int(t * sample_rate)
        end = min(int((t + note_sec) * sample_rate), audio_samples)

        t_arr = np.arange(end - start) / sample_rate
        audio[start:end] += 0.5 * np.sin(2 * np.pi * freq * t_arr)

        t += note_sec

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Convert to stereo tensor; cast to VAE dtype to avoid mismatch (B2)
    audio_tensor = torch.from_numpy(audio).float()
    audio_tensor = torch.stack([audio_tensor, audio_tensor], dim=0).unsqueeze(0)
    audio_tensor = audio_tensor.to(device, dtype=dtype)

    # Encode to latent
    with torch.no_grad():
        latent = pipe.vae.encode(audio_tensor).latent_dist.sample()
        # RMS normalize
        latent_rms = compute_rms(latent)
        latent = latent / (latent_rms + 1e-8)

    return latent.to(dtype)


def _apply_latent_fusion(latents, control_latent, pipe, step_idx, scale, outer_seed: int = 0):
    """Apply variance-preserving latent fusion."""
    B, C, T = latents.shape

    # Match control latent shape
    ctrl = control_latent
    if ctrl.shape[0] != B:
        ctrl = ctrl.expand(B, -1, -1)
    if ctrl.shape[-1] != T:
        ctrl = F.interpolate(ctrl, size=T, mode='linear', align_corners=False)

    ctrl = ctrl.to(latents.device, latents.dtype)

    # Get sigma for noise matching
    sigmas = getattr(pipe.scheduler, 'sigmas', None)
    if sigmas is not None:
        idx = min(step_idx, len(sigmas) - 2)
        sigma = float(sigmas[idx])
    else:
        sigma = 1.0

    # Combine step and user seed so different --seed runs give different noise (O6)
    alpha = 1.0 / math.sqrt(1.0 + sigma ** 2)
    gen = torch.Generator(device=ctrl.device).manual_seed(step_idx + outer_seed * 9973)
    noise = torch.randn(ctrl.shape, generator=gen, dtype=ctrl.dtype, device=ctrl.device)
    ctrl_noisy = alpha * ctrl + sigma * noise

    # RMS normalize both
    latent_rms = compute_rms(latents)
    ctrl_rms = compute_rms(ctrl_noisy)

    x_n = latents / (latent_rms + 1e-8)
    z_n = ctrl_noisy / (ctrl_rms + 1e-8)

    # Variance-preserving fusion
    lam = min(max(scale, 0.0), 0.95)
    mixed = math.sqrt(1 - lam ** 2) * x_n + lam * z_n

    # Scale back
    with torch.no_grad():
        latents.copy_(mixed * latent_rms)


def _apply_probe_guidance(latents, probe, target_pc, scale, max_ratio):
    """Apply probe gradient guidance."""
    latents_f32 = latents.float()
    with torch.enable_grad():
        latents_for_grad = latents_f32.detach().clone().requires_grad_(True)
        logits = probe(latents_for_grad)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target_pc.expand(latents.shape[0], -1, -1),
            reduction='mean'
        )
        grad = torch.autograd.grad(loss, latents_for_grad)[0]  # float32

    # RMS normalize in float32
    grad_rms = compute_rms(grad)
    latent_rms = compute_rms(latents_f32)
    normalized_grad = grad / (grad_rms + 1e-8)

    update = scale * latent_rms * normalized_grad
    max_update = max_ratio * latent_rms
    update = torch.clamp(update, -max_update, max_update)

    with torch.no_grad():
        latents.sub_(update.to(latents.dtype))


if __name__ == "__main__":
    print("Inference module loaded successfully")
