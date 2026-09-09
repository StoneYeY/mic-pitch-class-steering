#!/usr/bin/env bash
# Exp B FULL: 15 test prompts x 5 melodies x 3 seeds x 9 methods (~2025 gens). Resumable.
set -o pipefail
export SPLIT=test SEEDS=0,1,2
export EXPA_DIR=results/003_expA_trajectories EXPC_DIR=results/004_expC_burst
export REF_DIR="$HOME/Desktop/MIC/outputs/reference_audio"
python w2s/scripts/expB_windows.py
