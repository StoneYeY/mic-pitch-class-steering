#!/usr/bin/env bash
# Rescore Exp C CLAP from saved wavs (runs after the full Exp B).
set -o pipefail
python w2s/scripts/rescore_expC_clap.py 2>&1 | tee "$W2S_RESULTS/rescore.txt"
