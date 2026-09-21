#!/usr/bin/env bash
# Job 024: random-direction control on Stable Audio 3 (medium-base): 3-step burst at position 10 (inside the
# early plateau, steps 4-18) on the 50 dev trials, random direction of identical update norm. Compare with
# results/020c_sa3_expC (probe gradient).
set -o pipefail
export W2S_BACKEND=sa3 SA3_PROBE=results/018b_sa3_probe/sa3_probe_best.pt W2S_CLAP=0
export W2S_RANDOM_GRAD=1 EXPC_POSITIONS=10 EXPC_LAMBDA=0.10
conda run -n stablenew --no-capture-output python w2s/scripts/expC_burst.py
