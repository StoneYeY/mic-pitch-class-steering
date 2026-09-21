#!/usr/bin/env bash
# Job 026: the schedule of [8] with the ORIGINAL target placement (target_fps=204.8, the legacy bug) on the full
# test set (15 prompts x 5 melodies x 3 seeds), evaluated against the true melody -> the "before correction" number
# for Sec. 3. Compare with results/009_expB_full (mlsp = corrected placement, same trials).
set -o pipefail
export SPLIT=test SEEDS=0,1,2 METHODS=mlsp K=15 LAMBDA=0.05 TARGET_FPS=204.8
python w2s/scripts/expB_windows.py
