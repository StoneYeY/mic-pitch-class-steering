#!/usr/bin/env bash
# Job 029: demo clips for the fixed-window conditions (Early 0-14 / Mid 18-32 / Late 35-49, K=15) of the SAO
# Exp B run, same six prompt-melody pairs as job 013, so the window ablation of Table 1 is audible on the demo page.
set -o pipefail
export EXPB_RUNS=runs/009_expB_full
export DEMO_CONDS=early,mid,late
python w2s/scripts/collect_demo.py
