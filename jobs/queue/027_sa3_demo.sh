#!/usr/bin/env bash
# Job 027: demo clips from the SA3 Exp B wavs (prompts 0,2,4,6; sao / mlsp / rapg_cal_dc) for the listening page.
set -o pipefail
export EXPB_RUNS=runs/021_sa3_expB
python w2s/scripts/collect_demo.py
