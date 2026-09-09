#!/usr/bin/env python
"""
Step 3: Run probe-guided music generation.

This script demonstrates Contribution 2:
Using trained probes as learned loss functions for
inference-time latent steering.

Usage:
    python scripts/03_run_inference.py --probe_path checkpoints/probe_best.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import soundfile as sf
from diffusers import StableAudioPipeline

import config
from src.probe import create_probe
from src.inference import ProbeGuidedGenerator, make_combined_callback


# Test melodies (MIDI pitch numbers)
TEST_MELODIES = {
    "ascending": [53, 55, 56, 58, 60, 61, 63, 65],
    "descending": [65, 63, 61, 60, 58, 56, 55, 53],
    "alternating": [53, 56, 60, 65, 60, 56, 53],
}

# Test prompts — 9 piano-containing prompts from the musicpref human preference dataset
# (musicprefs/human_preference.csv; all 9 unique prompts matching keyword "piano")
TEST_PROMPTS = [
    "Fast-paced Western Classical music with lively violin harmony, electric cello, piano and string accompaniments that are colourful, well-layered, dense, rich, full, pleasant, cheerful, merry and elegant.",
    "An alternative/indie song with groovy bass, punchy drums, piano chords, and synth melodies that create an addictive and retro sound.",
    "Classical music piece with a gentle piano tune and a theremin playing the main melody, creating a unique and heart-touching atmosphere suitable for an animation movie/TV series.",
    "Upbeat indie rock instrumental with piano melody, simple percussion, distortion guitar, and bass.",
    "A slow and captivating piano piece that evokes sombre and reflective emotions with simple yet effective motifs.",
    "Dramatic contemporary classical piano piece with accentuated playing style suitable for documentary or mystery/horror video game soundtrack.",
    "A piano arpeggio melody with reverb and environmental sounds, suitable for amateur video intros or outros.",
    "Relaxing instrumental with piano and French horn, featuring an ascending progression and arpeggiated chords.",
    "A melodic and emotional duet with piano, electric guitar, drums, and synthesizer arrangements that could be suitable for children.",
]


def main():
    parser = argparse.ArgumentParser(description="Run probe-guided generation")
    parser.add_argument(
        "--probe_path",
        type=str,
        default=str(config.CHECKPOINT_DIR / "probe_best.pt"),
        help="Path to trained probe checkpoint"
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="cnn",
        choices=["linear", "cnn"],
        help="Probe architecture type"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR / "generated_audio"),
        help="Output directory for generated audio"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.05,
        help="Probe guidance scale (start small, e.g., 0.01-0.1)"
    )
    parser.add_argument(
        "--guidance_start",
        type=float,
        default=0.4,
        help="Start guidance at this fraction of denoising (0.4 = last 60%)"
    )
    parser.add_argument(
        "--audio_length",
        type=float,
        default=5.0,
        help="Output audio length in seconds"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (seed for run i = seed + i)"
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="Number of seeds to run per (prompt, melody, method). "
             "Files are named prompt{N}_s{seed}_{melody}_{method}.wav. "
             "Existing files are skipped so runs can be resumed."
    )
    parser.add_argument(
        "--compare_methods",
        action="store_true",
        help="Generate with all methods for comparison"
    )
    parser.add_argument(
        "--verbose_guidance",
        action="store_true",
        help="Print probe_loss and update_rms at each guidance step (useful for verifying callback fires)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MIC - PROBE-GUIDED GENERATION")
    print("=" * 60)
    print(f"Probe path: {args.probe_path}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Guidance start: {args.guidance_start}")
    print(f"Audio length: {args.audio_length}s")
    print(f"Seeds: {args.n_seeds}  (base={args.seed})")
    print(f"Compare methods: {args.compare_methods}")
    total = len(TEST_PROMPTS) * len(TEST_MELODIES) * args.n_seeds
    if args.compare_methods:
        total *= 7
    print(f"Max files to generate: {total}")
    print("=" * 60)

    # Setup
    config.setup_memory_efficient()
    device = config.get_device()
    print(f"\nDevice: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("\nLoading Stable Audio Open...")
    pipe = StableAudioPipeline.from_pretrained(
        config.MODEL_ID,
        torch_dtype=config.DTYPE
    )
    pipe = pipe.to(device)
    print("Pipeline loaded!")

    # Load probe
    print(f"\nLoading probe from {args.probe_path}...")
    probe = create_probe(
        probe_type=args.probe_type,
        in_channels=config.PROBE_IN_CHANNELS,
        hidden_channels=config.PROBE_HIDDEN_CHANNELS,
        out_classes=config.PROBE_OUT_CLASSES,
    )
    checkpoint = torch.load(args.probe_path, map_location=device, weights_only=True)
    probe.load_state_dict(checkpoint['model_state_dict'])
    probe = probe.to(device)
    probe.eval()
    print(f"Probe loaded (epoch {checkpoint.get('epoch', 'N/A')})")

    # Create generator
    generator = ProbeGuidedGenerator(pipe=pipe, probe=probe, device=str(device))

    # Generate samples
    print("\n" + "=" * 60)
    print("GENERATING SAMPLES")
    print("=" * 60)

    seeds = [args.seed + i for i in range(args.n_seeds)]

    for prompt_idx, prompt in enumerate(TEST_PROMPTS):
        for melody_name, melody_notes in TEST_MELODIES.items():
            print(f"\nPrompt {prompt_idx+1}: '{prompt[:60]}...'")
            print(f"Melody: {melody_name} ({melody_notes[:4]}...)")

            if args.compare_methods:
                methods = [
                    ("baseline",    "none",     0.0),
                    ("itc",         "itc",      0.0),   # latent fusion only (no probe)
                    ("probe_low",   "probe",    0.03),
                    ("probe_mid",   "probe",    0.05),
                    ("probe_high",  "probe",    0.1),
                    ("probe_strong","probe",    0.2),
                    ("combined",    "combined", 0.05),
                ]
            else:
                methods = [("probe", "probe", args.guidance_scale)]

            for seed in seeds:
                for method_name, method_type, scale in methods:
                    # File naming: multi-seed runs use _s{seed}_ infix so each
                    # (prompt, melody, seed) triple is a distinct evaluation trial.
                    if args.n_seeds > 1:
                        filename = f"prompt{prompt_idx+1}_s{seed}_{melody_name}_{method_name}.wav"
                    else:
                        filename = f"prompt{prompt_idx+1}_{melody_name}_{method_name}.wav"
                    filepath = output_dir / filename

                    if filepath.exists():
                        print(f"  [skip] {filename}")
                        continue

                    print(f"  seed={seed} method={method_name}...", end=" ", flush=True)

                    if method_type == "probe":
                        audio = generator.generate(
                            prompt=prompt,
                            target_notes=melody_notes,
                            tempo=120.0,
                            note_duration_beats=1.0,
                            audio_length=args.audio_length,
                            probe_guidance_scale=scale,
                            probe_guidance_start=args.guidance_start,
                            seed=seed,
                            verbose=args.verbose_guidance,
                        )
                    elif method_type == "itc":
                        cb = make_combined_callback(
                            pipe=pipe,
                            probe=probe,
                            target_notes=melody_notes,
                            use_latent_fusion=True,
                            fusion_scale=0.1,
                            fusion_start=0.5,
                            fusion_every_n=5,
                            use_probe_guidance=False,
                            tempo=120.0,
                            note_duration_beats=1.0,
                            audio_length=args.audio_length,
                            num_inference_steps=50,
                            seed=seed,
                        )
                        output = pipe(
                            prompt=prompt,
                            negative_prompt="",
                            num_inference_steps=50,
                            guidance_scale=4.0,
                            audio_end_in_s=args.audio_length,
                            generator=torch.Generator(device).manual_seed(seed),
                            callback=cb,
                            callback_steps=1,
                        )
                        audio = output.audios[0].T.float().cpu().numpy()
                    elif method_type == "combined":
                        cb = make_combined_callback(
                            pipe=pipe,
                            probe=probe,
                            target_notes=melody_notes,
                            use_latent_fusion=True,
                            fusion_scale=0.1,
                            fusion_start=0.5,
                            fusion_every_n=5,
                            use_probe_guidance=True,
                            probe_guidance_scale=scale,
                            probe_guidance_start=args.guidance_start,
                            tempo=120.0,
                            note_duration_beats=1.0,
                            audio_length=args.audio_length,
                            num_inference_steps=50,
                            seed=seed,
                        )
                        output = pipe(
                            prompt=prompt,
                            negative_prompt="",
                            num_inference_steps=50,
                            guidance_scale=4.0,
                            audio_end_in_s=args.audio_length,
                            generator=torch.Generator(device).manual_seed(seed),
                            callback=cb,
                            callback_steps=1,
                        )
                        audio = output.audios[0].T.float().cpu().numpy()
                    else:
                        output = pipe(
                            prompt=prompt,
                            negative_prompt="",
                            num_inference_steps=50,
                            guidance_scale=4.0,
                            audio_end_in_s=args.audio_length,
                            generator=torch.Generator(device).manual_seed(seed),
                        )
                        audio = output.audios[0].T.float().cpu().numpy()

                    sf.write(str(filepath), audio, config.SAMPLE_RATE)
                    print(f"saved → {filename}")

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total files: {len(list(output_dir.glob('*.wav')))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
