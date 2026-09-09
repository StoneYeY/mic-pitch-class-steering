#!/usr/bin/env python
"""
Step 5: Compute FAD (Fréchet Audio Distance) using OpenL3 @ 44.1kHz stereo,
via Stability AI's stable-audio-metrics implementation.

This matches the protocol used in the ITC reference notebook (Cell 39) and the
Stable Audio paper, so the resulting FAD values are directly comparable to
prior reported numbers.

Why OpenL3 instead of PANN:
  - Frame-level embeddings (~9 per 5s clip @ hop=0.5s) — n_samples is
    n_files × ~9, not n_files. With 27 files per method we get ~243 frame
    embeddings → covariance estimation is well-behaved (vs. 27 file-pooled
    points in 2048-dim space with PANN, which produces numerically degenerate
    FAD values around 1e-5).
  - 1024-dim stereo embeddings (concat of L+R 512-dim) vs PANN's 2048-dim.
  - Peak-normalized loudness and soxr resampling matches the published
    protocol used by ITC and Stable Audio.

Pre-requisites:
  1) Clone stable-audio-metrics next to MIC (or anywhere; pass --sam_root):
       cd /home/stoneyey/Desktop && \
         git clone https://github.com/Stability-AI/stable-audio-metrics
  2) Install OpenL3 (pulls in TensorFlow 2.x as a transitive):
       conda activate mic
       pip install "openl3==0.4.2" "tensorflow==2.13.1" \
                   "kapre==0.3.7" "soxr" "pyloudnorm"
     (You can `pip install -r ../stable-audio-metrics/requirements.txt`
      instead, but it pins many other packages that may conflict with the
      mic env.)

Usage:
    python scripts/05_fad.py
    python scripts/05_fad.py --hop_size 1.0 --channels 1   # faster
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

# TensorFlow (used by OpenL3) tries to use the GPU but may fail on newer CUDA
# drivers with PTX incompatibility. Force CPU-only for OpenL3 inference so the
# GPU stays available for PyTorch / Stable Audio.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import librosa
import soundfile as sf

import config


# Method groups for per-method FAD reporting. Each method has its own row;
# the "_all_*" rollups give a sanity check by combining methods.
FAD_METHODS = ["baseline", "itc", "probe_low", "probe_mid", "probe_high",
               "probe_strong", "combined"]
FAD_GROUPS = {
    "_all_probe": ["probe_low", "probe_mid", "probe_high", "probe_strong"],
    "_all_generated": FAD_METHODS,
}


def build_reference_set(maestro_dir: Path, output_dir: Path,
                        n_clips: int = 150, clip_duration: float = 5.0,
                        target_sr: int = 44100, seed: int = 42) -> int:
    """Extract random non-overlapping clips from MAESTRO WAVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("ref_*.wav"))
    if len(existing) >= n_clips:
        print(f"Reference set already built: {len(existing)} clips in {output_dir}")
        return len(existing)

    rng = random.Random(seed)
    wav_files = sorted(maestro_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files in {maestro_dir}")

    print(f"Building reference set from {len(wav_files)} MAESTRO files "
          f"(target={n_clips} × {clip_duration}s clips)...")

    clip_samples = int(clip_duration * target_sr)
    clips_per_file = max(2, (n_clips + len(wav_files) - 1) // len(wav_files))
    clip_idx = 0

    for wav_path in wav_files:
        if clip_idx >= n_clips:
            break
        try:
            audio, sr = librosa.load(str(wav_path), sr=target_sr, mono=False)
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            total = audio.shape[1]
            if total < clip_samples:
                continue
            n_possible = total // clip_samples
            n_take = min(clips_per_file, n_possible, n_clips - clip_idx)
            positions = list(range(n_possible))
            rng.shuffle(positions)
            for pos in positions[:n_take]:
                start = pos * clip_samples
                chunk = audio[:, start:start + clip_samples].T
                sf.write(str(output_dir / f"ref_{clip_idx:04d}.wav"),
                         chunk, target_sr)
                clip_idx += 1
        except Exception as e:
            print(f"  Skipping {wav_path.name}: {e}")

    print(f"Reference set: {clip_idx} clips → {output_dir}")
    return clip_idx


def collect_files(audio_dir: Path, methods) -> list:
    """Collect WAVs whose filename ends with any of the given method tags."""
    files = []
    for m in methods:
        files.extend(sorted(audio_dir.glob(f"*_{m}.wav")))
    return sorted(set(files))


def stage_files(files, dest_dir: Path):
    """Copy a list of files into a clean destination directory."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    for f in files:
        shutil.copy(str(f), str(dest_dir / f.name))


def main():
    parser = argparse.ArgumentParser(description="OpenL3 FAD via stable-audio-metrics")
    parser.add_argument("--audio_dir", type=str,
                        default=str(config.OUTPUT_DIR / "generated_audio"))
    parser.add_argument("--maestro_dir", type=str,
                        default=str(config.DATA_DIR / "maestro" / "2018"))
    parser.add_argument("--reference_dir", type=str,
                        default=str(config.OUTPUT_DIR / "reference_audio"))
    parser.add_argument("--output_file", type=str,
                        default=str(config.OUTPUT_DIR / "evaluation" / "fad_results.json"))
    parser.add_argument("--n_reference", type=int, default=1000)
    parser.add_argument("--sam_root", type=str,
                        default=str(Path.home() / "Desktop" / "stable-audio-metrics"),
                        help="Path to cloned stable-audio-metrics repo")
    parser.add_argument("--samplingrate", type=int, default=44100,
                        help="Eval bandwidth (44100 or 48000). Stable Audio paper uses 48000.")
    parser.add_argument("--channels", type=int, default=2, choices=[1, 2],
                        help="2 = stereo (1024-dim); 1 = mono (512-dim, faster).")
    parser.add_argument("--hop_size", type=float, default=0.5,
                        help="OpenL3 hop_size in seconds (smaller → more frames).")
    parser.add_argument("--content_type", type=str, default="music",
                        choices=["music", "env"])
    parser.add_argument("--batching", type=int, default=16,
                        help="OpenL3 batch size (0 = no batching).")
    parser.add_argument("--ref_stats_path", type=str,
                        default=str(config.OUTPUT_DIR / "evaluation"
                                    / "_openl3_ref_stats.npz"),
                        help="Cached reference μ/Σ. Recomputed only if missing.")
    parser.add_argument("--rebuild_ref_stats", action="store_true",
                        help="Force re-extraction of reference embeddings.")
    args = parser.parse_args()

    # Wire stable-audio-metrics into the import path
    sam_root = Path(args.sam_root)
    if not (sam_root / "src" / "openl3_fd.py").exists():
        raise FileNotFoundError(
            f"stable-audio-metrics not found at {sam_root}. "
            f"Clone it with:\n  cd {sam_root.parent} && "
            f"git clone https://github.com/Stability-AI/stable-audio-metrics"
        )
    sys.path.insert(0, str(sam_root))
    from src.openl3_fd import (
        extract_embeddings, extract_embeddings_nobatching,
        calculate_embd_statistics, calculate_frechet_distance,
    )

    audio_dir = Path(args.audio_dir)
    reference_dir = Path(args.reference_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ref_stats_path = Path(args.ref_stats_path)

    print("=" * 72)
    print("MIC - OpenL3 FAD  (stable-audio-metrics)")
    print("=" * 72)
    print(f"Eval bandwidth: {args.samplingrate} Hz  |  channels: {args.channels}  "
          f"({1024 if args.channels == 2 else 512}-dim)")
    print(f"OpenL3 hop_size: {args.hop_size}s  |  content_type: {args.content_type}")
    print(f"Reference: {args.n_reference} MAESTRO clips × 5s @ 44100Hz")
    print("=" * 72)

    # Build reference WAV set if missing
    n_ref = build_reference_set(
        maestro_dir=Path(args.maestro_dir),
        output_dir=reference_dir,
        n_clips=args.n_reference,
    )

    def extract(directory_glob):
        if args.batching:
            return extract_embeddings(directory_glob, args.channels,
                                      args.samplingrate, args.content_type,
                                      args.hop_size, batch_size=args.batching)
        return extract_embeddings_nobatching(directory_glob, args.channels,
                                             args.samplingrate, args.content_type,
                                             args.hop_size)

    # Reference embedding stats: cache on disk; recompute only if missing or forced
    if ref_stats_path.exists() and not args.rebuild_ref_stats:
        print(f"\n[REF STATS] Loading cached: {ref_stats_path}")
        loaded = np.load(ref_stats_path)
        mu_ref, sigma_ref = loaded["mu_ref"], loaded["sigma_ref"]
        n_ref_frames = int(loaded["n_frames"]) if "n_frames" in loaded.files else -1
    else:
        print(f"\n[REF STATS] Extracting reference embeddings "
              f"(this is the slow step; cached to {ref_stats_path.name}).")
        ref_emb_list = extract(str(reference_dir / "*.wav"))
        n_ref_frames = sum(e.shape[0] if e.ndim > 1 else 1 for e in ref_emb_list) \
                       if isinstance(ref_emb_list, list) else len(ref_emb_list)
        mu_ref, sigma_ref = calculate_embd_statistics(ref_emb_list)
        ref_stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ref_stats_path, mu_ref=mu_ref, sigma_ref=sigma_ref,
                 n_frames=n_ref_frames, n_files=n_ref,
                 channels=args.channels, samplingrate=args.samplingrate,
                 hop_size=args.hop_size, content_type=args.content_type)
        print(f"[REF STATS] Saved: {ref_stats_path}  ({n_ref_frames} frames)")

    # Reference split-half floor (frame-level — meaningful with OpenL3)
    print("\n[FLOOR] Computing reference split-half FAD (lower bound)...")
    ref_files = sorted(reference_dir.glob("ref_*.wav"))
    rng = random.Random(0)
    shuffled = ref_files.copy()
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    tmp_root = Path(tempfile.mkdtemp(prefix="mic_fad_openl3_"))
    half_a, half_b = tmp_root / "ref_half_a", tmp_root / "ref_half_b"
    stage_files(shuffled[:mid], half_a)
    stage_files(shuffled[mid:], half_b)
    emb_a = extract(str(half_a / "*.wav"))
    emb_b = extract(str(half_b / "*.wav"))
    mu_a, sigma_a = calculate_embd_statistics(emb_a)
    mu_b, sigma_b = calculate_embd_statistics(emb_b)
    floor = float(calculate_frechet_distance(mu_a, sigma_a, mu_b, sigma_b))
    print(f"[FLOOR] Reference split-half FAD = {floor:.4f}")

    # Per-method FAD
    results = {
        "_reference_split_half_floor": {
            "fad_openl3": floor, "n_files_per_half": mid,
            "note": "Lower bound: FAD between two random halves of the MAESTRO reference.",
        }
    }

    print(f"\n{'Method':<15} | {'N files':<8} | FAD (OpenL3)")
    print("-" * 50)
    for method in FAD_METHODS:
        files = collect_files(audio_dir, [method])
        if not files:
            print(f"{method:<15} | {'—':<8} | (no files)")
            continue
        method_dir = tmp_root / method
        stage_files(files, method_dir)
        try:
            eval_emb = extract(str(method_dir / "*.wav"))
            mu_e, sigma_e = calculate_embd_statistics(eval_emb)
            fad = float(calculate_frechet_distance(mu_e, sigma_e, mu_ref, sigma_ref))
            n_frames_eval = sum(e.shape[0] if e.ndim > 1 else 1 for e in eval_emb) \
                            if isinstance(eval_emb, list) else len(eval_emb)
            results[method] = {
                "fad_openl3": fad,
                "n_files": len(files),
                "n_frames": n_frames_eval,
                "fad_above_floor": fad - floor,
            }
            print(f"{method:<15} | {len(files):<8} | {fad:.4f}  (Δfloor={fad - floor:+.4f})")
        except Exception as e:
            print(f"{method:<15} | {len(files):<8} | ERROR: {e}")
            results[method] = {"fad_openl3": None, "n_files": len(files),
                               "error": str(e)}

    # Group rollups (sanity: probe_all should bracket individual probe scales)
    for group_name, methods in FAD_GROUPS.items():
        files = collect_files(audio_dir, methods)
        if not files:
            continue
        group_dir = tmp_root / group_name
        stage_files(files, group_dir)
        try:
            eval_emb = extract(str(group_dir / "*.wav"))
            mu_e, sigma_e = calculate_embd_statistics(eval_emb)
            fad = float(calculate_frechet_distance(mu_e, sigma_e, mu_ref, sigma_ref))
            results[group_name] = {
                "fad_openl3": fad, "n_files": len(files),
                "methods_included": methods,
                "fad_above_floor": fad - floor,
            }
            print(f"{group_name:<15} | {len(files):<8} | {fad:.4f}  "
                  f"(Δfloor={fad - floor:+.4f})")
        except Exception as e:
            results[group_name] = {"fad_openl3": None, "error": str(e),
                                   "methods_included": methods}

    shutil.rmtree(tmp_root, ignore_errors=True)

    fad_report = {
        "model": f"OpenL3 mel256 / {args.content_type} / "
                 f"{1024 if args.channels == 2 else 512}-dim "
                 f"({'stereo' if args.channels == 2 else 'mono'})",
        "samplingrate_hz": args.samplingrate,
        "openl3_hop_size_sec": args.hop_size,
        "reference": f"{n_ref} MAESTRO clips × 5s @ 44100Hz "
                     f"({n_ref_frames if n_ref_frames > 0 else '?'} frame-embeddings)",
        "implementation": "Stability-AI/stable-audio-metrics src/openl3_fd.py "
                          "(matches ITC paper protocol).",
        "results": results,
    }
    with open(output_file, "w") as f:
        json.dump(fad_report, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("=" * 72)
    print("Sanity check: all method FADs should be > floor and within 1-2 OoM "
          "of each other.")
    print("=" * 72)


if __name__ == "__main__":
    main()
