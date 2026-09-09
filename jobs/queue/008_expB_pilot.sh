#!/usr/bin/env bash
# Exp B PILOT: 3 test prompts x 5 melodies x 1 seed x 9 methods (~135 gens) — Gate 2 + CLAP check.
set -o pipefail
export SPLIT=test PROMPTS=0,1,2 SEEDS=0
export EXPA_DIR=results/003_expA_trajectories EXPC_DIR=results/004_expC_burst
export REF_DIR="$HOME/Desktop/MIC/outputs/reference_audio"
python w2s/scripts/expB_windows.py
