# Shuffled-target control — results and interpretation

For: writing the reviewer-RPZP8 rebuttal paragraph, the Table 2 row, and the
Wilcoxon statement in the MLSP 2026 camera-ready of *"Pitch-Class Steering for
Diffusion-Based Music Generation via Latent-Space Probes"*.

Run 2026-08-17 on the generation machine (`~/Desktop/MIC`), librosa 0.11.0,
scipy 1.13.1, numpy 1.26.4. The sanity check passed: the pipeline reproduces
the paper's Table 2 values exactly (λ=0.05 → 0.274, baseline → 0.112), so
these numbers are directly comparable to the paper's.

## The experiment

Each of the 189 generated clips (9 prompts × 3 melodies × 7 conditions) was
scored with the project's own melodic-coherence evaluation twice: against its
true time-varying pitch-class target, and against all 11 non-trivial circular
rotations of that target (the shuffled control). Rotation changes only
pitch-class identity; rhythm, active-frame mask, and the metric's denominator
are identical. The shuffled mean is therefore the metric's *empirical* chance
level on this exact audio. Per condition, a one-sided Wilcoxon signed-rank test
pairs each clip's true score with its own shuffled mean. No trials were dropped
by the 20% voiced-coverage rule (all 189 valid).

## Results (full console output is in console_output.txt)

| condition   | n  | true (mean ± sd) | shuffled (mean ± sd) | ratio | p (one-sided Wilcoxon) |
|-------------|----|------------------|----------------------|-------|------------------------|
| baseline    | 27 | 0.112 ± 0.104    | 0.081 ± 0.009        | 1.38  | 0.14                   |
| itc         | 27 | 0.122 ± 0.103    | 0.080 ± 0.009        | 1.53  | 0.089                  |
| probe_0.03  | 27 | 0.236 ± 0.104    | 0.069 ± 0.009        | 3.39  | 1.9e-07                |
| probe_0.05  | 27 | 0.274 ± 0.084    | 0.066 ± 0.008        | 4.15  | 7.5e-09                |
| probe_0.1   | 27 | 0.266 ± 0.071    | 0.067 ± 0.006        | 4.00  | 7.5e-09                |
| probe_0.2   | 27 | 0.273 ± 0.064    | 0.066 ± 0.006        | 4.12  | 7.5e-09                |
| combined    | 27 | 0.276 ± 0.118    | 0.066 ± 0.011        | 4.19  | 1.4e-07                |

`probe_0.05` is the paper's λ=0.05 condition (filename method `probe_mid`;
λ values per `scripts/03_run_inference.py`: low=0.03, mid=0.05, high=0.1,
strong=0.2). Per-trial numbers, including per-clip shuffled max, are in
`shuffled_results.csv`.

## Interpretation — what the rebuttal can say

1. **The reviewer's intuition about the baseline is correct, and that helps
   us.** The unguided baseline (0.112) is statistically indistinguishable from
   its own shuffled control (0.081, p=0.14): unguided generation carries
   essentially no information about the target melody. The same holds for ITC
   alone (p=0.089).

2. **The guided conditions sit far above empirical chance.** At λ=0.05, true
   coherence is 0.274 vs an empirical chance level of 0.066 — a 4.15× ratio,
   Wilcoxon one-sided p = 7.5×10⁻⁹, n = 27. Every probe-guided condition and
   the combined condition show the same pattern (ratios 3.4–4.2, p ≤ 1.9×10⁻⁷).

3. **The empirical chance level is at or below 1/12, not above it.** The
   shuffled level is ≈0.066–0.081, bracketing/undercutting the naive 1/12 ≈
   0.083. So the reported gains are not inflated by a too-low reference point;
   if anything the naive 1/12 slightly *overstates* chance for the guided
   clips. (Chance is lower for guided clips because guidance concentrates
   voiced frames on the target pitch class, so a rotated target matches less
   often than a uniform-random reference would.)

4. Suggested framing: the probe-guidance gain over baseline (0.274 vs 0.112)
   is not a small shift within noise around chance — guidance moves the
   metric from a level indistinguishable from chance to >4× chance, with the
   chance level measured empirically on the same audio, same rhythm, same
   denominator.

## Method details (for the reproducibility statement)

- Scoring reused the project's own evaluation: the `--impl` adapter
  (`mic_eval_adapter.py`, included) reproduces `compute_melody_coherence` from
  `scripts/04_evaluate.py` step for step (native-sr load; pYIN C2–C7 with
  hop = int(sr/86); frames-per-second derived from measured audio duration;
  monophonic per-frame target one-hot encoded to (n_frames, 12); voiced-coverage
  denominator identical). The handoff script's rotation and statistics logic
  was not modified.
- MELODIES actually used (from `scripts/03_run_inference.py` `TEST_MELODIES`,
  identical in `04_evaluate.py`) — all three overridden via `--impl`:

  ```python
  "ascending":   [53, 55, 56, 58, 60, 61, 63, 65],
  "descending":  [65, 63, 61, 60, 58, 56, 55, 53],
  "alternating": [53, 56, 60, 65, 60, 56, 53],
  ```

## ⚠ One correction needed in the camera-ready text

The paper text (per the handoff instructions) describes ascending/descending as
the **F-major** scale F3→F4. The generation code actually uses
F–G–A♭–B♭–C–D♭–E♭–F — the **F natural-minor** scale — and the alternating
arpeggio is F–A♭–C–F (F minor). The evaluation and all numbers above use the
melodies the code really generated, so the results are internally consistent;
only the prose description of the scale in the paper needs fixing.

## Files in this package

| file | contents |
|---|---|
| `shuffled_results.csv` | per-trial results: true coherence, shuffled mean/max, voiced fraction |
| `console_output.txt` | complete verbatim console output incl. SANITY CHECK line |
| `trials.csv` | manifest (clip → melody, condition, prompt) |
| `mic_eval_adapter.py` | the `--impl` adapter module used |
| `SENDBACK.md` | checklist answers for the handoff's Step 6 |
