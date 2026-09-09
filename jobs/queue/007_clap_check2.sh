#!/usr/bin/env bash
set -o pipefail
python w2s/scripts/clap_check2.py 2>&1 | tee "$W2S_RESULTS/clap_check2.txt"
