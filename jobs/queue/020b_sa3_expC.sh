#!/usr/bin/env bash
# Job 020b: Exp C on Stable Audio 3 medium-base: 3-step burst at 12 positions x 50 dev trials, lambda=0.10 (as SAO),
# paired against the same-seed unguided run. CLAP off here (rescored later if needed).
set -o pipefail
export W2S_BACKEND=sa3 SA3_PROBE=results/018b_sa3_probe/sa3_probe_best.pt W2S_CLAP=0
conda run -n stablenew --no-capture-output python w2s/scripts/expC_burst.py
