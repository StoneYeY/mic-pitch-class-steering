#!/usr/bin/env bash
# Job 019: Exp A on Stable Audio 3 medium-base: 50 unguided 50-step trajectories (5 dev prompts x 10 seeds),
# probe micro-F1 vs pYIN pseudo-labels and target-free reliability R_t per step.
set -o pipefail
conda activate stablenew
export W2S_BACKEND=sa3 SA3_PROBE=results/018_sa3_probe/sa3_probe_best.pt
python w2s/scripts/expA_trajectories.py
