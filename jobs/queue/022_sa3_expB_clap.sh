#!/usr/bin/env bash
# Job 022: CLAP for the SA3 Exp B wavs (job 021 ran without CLAP); runs in the default (mic) env, no generation.
set -o pipefail
export EXPB_RES=results/021_sa3_expB EXPB_RUNS=runs/021_sa3_expB SPLIT=test
python w2s/scripts/rescore_expB_clap.py
