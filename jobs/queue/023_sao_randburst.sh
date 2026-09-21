#!/usr/bin/env bash
# Job 023: random-direction control on SAO. Same 3-step burst at the sensitivity peak (position 26) on the 50 dev
# trials, but the probe gradient is replaced by a random direction of identical update norm. Compare with
# results/004_expC_burst (probe gradient, same trials/seeds).
set -o pipefail
export W2S_RANDOM_GRAD=1 EXPC_POSITIONS=26 EXPC_LAMBDA=0.10
python w2s/scripts/expC_burst.py
