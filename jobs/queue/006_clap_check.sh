#!/usr/bin/env bash
set -o pipefail
python w2s/scripts/clap_check.py 2>&1 | tee "$W2S_RESULTS/clap_check.txt"
