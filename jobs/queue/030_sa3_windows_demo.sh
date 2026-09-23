#!/usr/bin/env bash
# Job 030: same as 029 for the Stable Audio 3 Exp B run (prompts 0,2,4,6 exist there).
set -o pipefail
export EXPB_RUNS=runs/021_sa3_expB
export DEMO_CONDS=early,mid,late
python w2s/scripts/collect_demo.py
