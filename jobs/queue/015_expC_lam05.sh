#!/usr/bin/env bash
# Exp C robustness (reviewer request): repeat the burst-intervention sensitivity profile at the
# guidance strength used in the main experiments (lambda = 0.05 instead of 0.10), same 3-step burst,
# same 12 positions x 50 dev trials. Checks that the mid-sampling peak of dC(t) does not depend on lambda.
set -o pipefail
export EXPC_LAMBDA=0.05
python w2s/scripts/expC_burst.py
