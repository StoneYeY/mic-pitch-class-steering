#!/usr/bin/env bash
# Exp A: probe reliability along real trajectories (50 unguided runs, 5 dev prompts x 10 seeds).
set -o pipefail
python w2s/scripts/expA_trajectories.py
