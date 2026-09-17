#!/usr/bin/env bash
# Job 021: small Exp B on Stable Audio 3 medium-base (paired 50 trials: 10 test prompts x 5 melodies x seed 0),
# fixed windows vs the schedule calibrated on the SA3 sensitivity profile (Exp C, job 020c). K=15, lambda=0.05.
set -o pipefail
export W2S_BACKEND=sa3 SA3_PROBE=results/018b_sa3_probe/sa3_probe_best.pt W2S_CLAP=0
export SPLIT=test SEEDS=0 PROMPTS=0,1,2,3,4,5,6,7,8,9 K=15 LAMBDA=0.05
export METHODS=sao,early,mid,late,uniform,mlsp,rapg_cal_dc,topk_dc_const
export EXPA_DIR=results/019c_sa3_expA EXPC_DIR=results/020c_sa3_expC
conda run -n stablenew --no-capture-output python w2s/scripts/expB_windows.py
