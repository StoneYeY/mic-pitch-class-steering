#!/usr/bin/env bash
set -o pipefail
python w2s/scripts/clap_debug.py 2>&1 | tee "$W2S_RESULTS/clap_debug5.txt"
