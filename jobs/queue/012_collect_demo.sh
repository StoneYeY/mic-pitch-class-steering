#!/usr/bin/env bash
set -o pipefail
export EXPB_RUNS=runs/009_expB_full
python w2s/scripts/collect_demo.py
