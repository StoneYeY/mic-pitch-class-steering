#!/usr/bin/env python
"""
Step 4: Evaluate generated audio.

Computes:
- Melody coherence (pitch class accuracy, voiced-coverage-gated)
- Per-trial pivot for paired statistical tests
- Wilcoxon signed-rank vs baseline for each method

Usage:
    python scripts/04_evaluate.py --audio_dir outputs/generated_audio
"""

import argparse
import sys
import math
import re
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import librosa
import json
from scipy import stats as sp_stats

import config


TEST_MELODIES = {
    "ascending": [53, 55, 56, 58, 60, 61, 63, 65],
    "descending": [65, 63, 61, 60, 58, 56, 55, 53],
    "alternating": [53, 56, 60, 65, 60, 56, 53],
}

# Fraction of target-region frames that must be voiced for the score to be valid.
# Below this threshold the coherence is unreliable (a handful of accidentally-matching
# frames can produce artificially high scores).
MIN_VOICED_COVERAGE = 0.20


def compute_melody_coherence(
    audio_path: str,
    target_notes: list,
    tempo: float = 120.0,
    note_duration_beats: float = 1.0,
    frame_rate: float = 86.0,
    min_coverage: float = MIN_VOICED_COVERAGE,
) -> dict:
    """
    Compute frame-wise pitch class agreement with voiced-coverage guard.

    Returns:
        coherence:       NaN when voiced_coverage < min_coverage, else [0, 1]
        voiced_coverage: voiced frames in target region / total target frames
        voiced_frames:   count of voiced frames that fell inside the target region
        target_frames:   total frames inside the target melody region
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    audio_duration = len(y) / sr

    hop_length = int(sr / frame_rate)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        hop_length=hop_length,
    )

    midi = librosa.hz_to_midi(f0)
    pc_gen = np.full(len(midi), -1, dtype=np.int32)
    voiced_idx = np.isfinite(midi)
    pc_gen[voiced_idx] = (np.round(midi[voiced_idx]).astype(int) % 12)

    n_frames = len(pc_gen)
    sec_per_beat = 60.0 / tempo
    note_sec = note_duration_beats * sec_per_beat
    fps = n_frames / audio_duration

    pc_target = np.full(n_frames, -1, dtype=np.int32)
    t = 0.0
    for note in target_notes:
        if t >= audio_duration:
            break
        start_frame = int(t * fps)
        end_frame = int((t + note_sec) * fps)
        end_frame = max(end_frame, start_frame + 1)
        end_frame = min(end_frame, n_frames)
        pc_target[start_frame:end_frame] = note % 12
        t += note_sec

    target_frames = int((pc_target != -1).sum())
    valid = (pc_target != -1) & (pc_gen != -1)
    voiced_frames = int(valid.sum())
    voiced_coverage = voiced_frames / max(1, target_frames)

    if voiced_coverage < min_coverage:
        coherence = float('nan')
    else:
        coherence = float((pc_target[valid] == pc_gen[valid]).mean())

    return {
        'coherence': coherence,
        'voiced_coverage': voiced_coverage,
        'voiced_frames': voiced_frames,
        'target_frames': target_frames,
    }


def parse_prompt_from_filename(filename: str):
    m = re.search(r'(prompt\d+)', filename)
    return m.group(1) if m else None


def parse_seed_from_filename(filename: str):
    """Return seed string if present (_s{seed}_ infix), else None."""
    m = re.search(r'_s(\d+)_', filename)
    return m.group(1) if m else None


def parse_melody_from_filename(filename: str):
    for name in TEST_MELODIES:
        if name in filename:
            return name
    return None


def clean_for_json(obj):
    """Replace float NaN with None (JSON null) recursively."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated audio")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=str(config.OUTPUT_DIR / "generated_audio"),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(config.OUTPUT_DIR / "evaluation" / "results.json"),
    )
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--note_duration", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 60)
    print("MIC - EVALUATION")
    print("=" * 60)
    print(f"Audio directory: {args.audio_dir}")
    print(f"Min voiced coverage: {MIN_VOICED_COVERAGE:.0%}")
    print("=" * 60)

    audio_dir = Path(args.audio_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"\nFound {len(audio_files)} audio files")

    # per_trial["{prompt}[_s{seed}]_{melody}"][method] = {coherence, ...}
    # Each (prompt, seed, melody) triple is an independent trial; seeded files
    # use the _s{seed}_ infix while legacy single-seed files have no infix.
    per_trial = {}
    # flat list per method for summary
    by_method = {}

    for audio_path in audio_files:
        filename = audio_path.name

        # Parse method — longest/most-specific substrings first to avoid partial matches
        method = "unknown"
        for m in ["baseline", "combined", "probe_low", "probe_mid", "probe_high",
                  "probe_strong", "probe", "itc"]:
            if m in filename:
                method = m
                break

        prompt = parse_prompt_from_filename(filename)
        melody = parse_melody_from_filename(filename)
        if melody is None or prompt is None:
            print(f"  Skipping {filename} (parse failed)")
            continue

        seed_str = parse_seed_from_filename(filename)
        trial_key = f"{prompt}_s{seed_str}_{melody}" if seed_str else f"{prompt}_{melody}"
        target_notes = TEST_MELODIES[melody]

        try:
            res = compute_melody_coherence(
                str(audio_path), target_notes,
                tempo=args.tempo, note_duration_beats=args.note_duration,
            )
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            res = {'coherence': float('nan'), 'voiced_coverage': 0.0,
                   'voiced_frames': 0, 'target_frames': 0}

        if trial_key not in per_trial:
            per_trial[trial_key] = {}
        per_trial[trial_key][method] = res

        if method not in by_method:
            by_method[method] = {'coherence': [], 'voiced_coverage': [], 'files': []}
        by_method[method]['coherence'].append(res['coherence'])
        by_method[method]['voiced_coverage'].append(res['voiced_coverage'])
        by_method[method]['files'].append(filename)

        coh_str = 'NaN' if math.isnan(res['coherence']) else f"{res['coherence']:.4f}"
        print(f"  {filename}: coherence={coh_str},"
              f" coverage={res['voiced_coverage']:.2f}"
              f" ({res['voiced_frames']}/{res['target_frames']} voiced frames)")

    # Summary (NaN-excluded)
    print("\n" + "=" * 60)
    print(f"RESULTS SUMMARY  (NaN = voiced coverage < {MIN_VOICED_COVERAGE:.0%})")
    print("=" * 60)

    summary = {}
    for method, data in by_method.items():
        scores = np.array(data['coherence'], dtype=float)
        valid_mask = ~np.isnan(scores)
        valid_scores = scores[valid_mask]
        n_valid = int(valid_mask.sum())
        n_total = len(scores)

        summary[method] = {
            'mean_coherence':   float(np.mean(valid_scores))   if n_valid > 0 else float('nan'),
            'std_coherence':    float(np.std(valid_scores))    if n_valid > 0 else float('nan'),
            'median_coherence': float(np.median(valid_scores)) if n_valid > 0 else float('nan'),
            'n_valid': n_valid,
            'n_total': n_total,
        }

        mean_s = f"{np.mean(valid_scores):.4f}" if n_valid > 0 else "NaN"
        std_s  = f"{np.std(valid_scores):.4f}"  if n_valid > 0 else "NaN"
        print(f"\n{method}:")
        print(f"  Melody Coherence: {mean_s} ± {std_s}  (n_valid={n_valid}/{n_total})")

    # Wilcoxon signed-rank vs baseline (E5)
    print("\n" + "=" * 60)
    print("WILCOXON SIGNED-RANK vs BASELINE  (one-sided: method > baseline)")
    print("=" * 60)

    wilcoxon_results = {}
    for method in ["itc", "probe_low", "probe_mid", "probe_high", "probe_strong", "combined"]:
        if method not in by_method:
            continue
        pairs = []
        for trial_key in sorted(per_trial):
            if 'baseline' not in per_trial[trial_key]:
                continue
            if method not in per_trial[trial_key]:
                continue
            b = per_trial[trial_key]['baseline']['coherence']
            p = per_trial[trial_key][method]['coherence']
            if not (math.isnan(b) or math.isnan(p)):
                pairs.append((p, b))

        n_pairs = len(pairs)
        if n_pairs < 3:
            print(f"{method:12s}: only {n_pairs} valid pairs — skipping test")
            wilcoxon_results[method] = {'n_pairs': n_pairs, 'skipped': True}
            continue

        diffs = [p - b for p, b in pairs]
        if all(d == 0 for d in diffs):
            print(f"{method:12s}: all diffs zero — skipping test")
            wilcoxon_results[method] = {'n_pairs': n_pairs, 'all_zero': True}
            continue

        try:
            W, pval = sp_stats.wilcoxon(diffs, alternative='greater', zero_method='wilcox')
            median_diff = float(np.median(diffs))
            # Effect size: r = Z / sqrt(n), where Z approximated from normal approx
            # For reporting, include W, p, median_diff
            sig = " *" if pval < 0.05 else ""
            wilcoxon_results[method] = {
                'W': float(W), 'p_value': float(pval),
                'median_diff': median_diff, 'n_pairs': n_pairs,
            }
            print(f"{method:12s}: W={W:.1f}, p={pval:.4f}{sig},"
                  f" median_diff={median_diff:+.4f}, n={n_pairs}")
        except Exception as e:
            print(f"{method:12s}: error — {e}")
            wilcoxon_results[method] = {'n_pairs': n_pairs, 'error': str(e)}

    # Save results
    full_results = {
        'summary': summary,
        'wilcoxon_vs_baseline': wilcoxon_results,
        'per_trial': {
            k: {m: v for m, v in sorted(v_dict.items())}
            for k, v_dict in sorted(per_trial.items())
        },
        'per_method_details': {
            m: {
                'files': d['files'],
                'coherence': d['coherence'],
                'voiced_coverage': d['voiced_coverage'],
            }
            for m, d in by_method.items()
        },
    }

    with open(output_file, 'w') as f:
        json.dump(clean_for_json(full_results), f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Method':<15} | {'Mean (valid)':<12} | {'Median':<8} | n_valid/n")
    print("-" * 55)
    for method in sorted(summary):
        s = summary[method]
        mean_s = f"{s['mean_coherence']:.4f}"   if not math.isnan(s['mean_coherence'])   else "NaN "
        med_s  = f"{s['median_coherence']:.4f}" if not math.isnan(s['median_coherence']) else "NaN "
        print(f"{method:<15} | {mean_s:<12} | {med_s:<8} | {s['n_valid']}/{s['n_total']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
