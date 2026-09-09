# Shuffled-target control — send-back package

Run date: 2026-08-17. Machine: the MIC generation machine
(`~/Desktop/MIC`), librosa 0.11.0, scipy 1.13.1, numpy 1.26.4.

## Files in this directory

| File | What it is |
|---|---|
| `shuffled_results.csv` | Per-trial results (189 trials = 27 clips × 7 conditions) |
| `console_output.txt` | Complete console output, verbatim |
| `mic_eval_adapter.py` | The `--impl` adapter module (see item 4) |
| `trials.csv` | The manifest used |

## 2. Complete console output (verbatim)

```
using build_target, melodic_coherence, analyze, MELODIES from mic_eval_adapter.py
no trials dropped by the voiced-fraction rule

condition          n              true          shuffled   ratio   p (one-sided)
baseline          27     0.112 ± 0.104     0.081 ± 0.009    1.38            0.14
combined          27     0.276 ± 0.118     0.066 ± 0.011    4.19         1.4e-07
itc               27     0.122 ± 0.103     0.080 ± 0.009    1.53           0.089
probe_0.03        27     0.236 ± 0.104     0.069 ± 0.009    3.39         1.9e-07
probe_0.05        27     0.274 ± 0.084     0.066 ± 0.008    4.15         7.5e-09
probe_0.1         27     0.266 ± 0.071     0.067 ± 0.006    4.00         7.5e-09
probe_0.2         27     0.273 ± 0.064     0.066 ± 0.006    4.12         7.5e-09

SANITY CHECK  probe_0.05: computed 0.274, paper reports 0.274  ->  MATCH
```

## 3. Final MELODIES dict actually used

Taken from the generation script (`scripts/03_run_inference.py`, `TEST_MELODIES`,
identical in `scripts/04_evaluate.py`):

```python
    "ascending":   [53, 55, 56, 58, 60, 61, 63, 65],
    "descending":  [65, 63, 61, 60, 58, 56, 55, 53],
    "alternating": [53, 56, 60, 65, 60, 56, 53],
```

NOTE: these differ from the handoff script's defaults. The generation code used
an F *harmonic-minor-flavored* scale, not the F-major scale the handoff assumed
for ascending/descending, and the alternating arpeggio is F–A♭–C–F (F minor).
All three entries were overridden via `--impl`, so the handoff placeholders were
never used.

## 4. `--impl` — yes, used

Command line:

```
python shuffled_target_baseline.py --manifest trials.csv \
  --impl mic_eval_adapter.py --out shuffled_results.csv
```

`mic_eval_adapter.py` (included) overrides all four hooks —
`MELODIES`, `analyze`, `build_target`, `melodic_coherence` — and reproduces
`compute_melody_coherence` from the project's own `scripts/04_evaluate.py`
step for step (native-sr load, hop = int(sr/86), pYIN C2–C7, fps derived from
actual audio duration, one-hot encoding of the original monophonic per-frame
target, identical voiced-coverage denominator). The rotation and statistics
logic of the handoff script was not modified.

Differences from the handoff reconstruction that made `--impl` necessary:
the original eval uses `sr=None` + `hop_length=int(sr/86)` (not fixed
SR/HOP_LENGTH constants), derives frames-per-second from the measured audio
duration, and enforces `end_frame >= start_frame + 1` per note.

Sanity: probe_0.05 mean 0.274 == paper's Table 2 value (MATCH); baseline mean
0.112 also reproduces the paper's 0.112. No trials were dropped by the 20%
voiced-fraction rule.

## Manifest condition mapping

Filename method → manifest `cond` (λ from `scripts/03_run_inference.py`):
`probe_low`→`probe_0.03`, `probe_mid`→`probe_0.05` (the paper's λ=0.05 set),
`probe_high`→`probe_0.1`, `probe_strong`→`probe_0.2`; `baseline`, `itc`,
`combined` unchanged. 27 clips per condition (9 prompts × 3 melodies), 189 total.
