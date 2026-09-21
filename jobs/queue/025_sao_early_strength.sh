#!/usr/bin/env bash
# Job 025: does the early window (steps 0-14) work on SAO if one pushes harder? lambda = alpha in {0.2, 0.5, 1.0}
# (the clamp alpha*rms(z) is raised together with lambda so it does not bind), 50 dev trials, K=15.
set -o pipefail
export SPLIT=dev SEEDS=0,1 METHODS=sao,early K=15
for A in 0.2 0.5 1.0; do
  echo "### lambda=alpha=$A"
  LAMBDA=$A W2S_ALPHA=$A W2S_RESULTS="$W2S_RESULTS/a$A" W2S_RUNS="$W2S_RUNS/a$A" python w2s/scripts/expB_windows.py || exit 1
done
