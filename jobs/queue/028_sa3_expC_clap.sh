#!/usr/bin/env bash
# Job 028: CLAP for the SA3 Exp C wavs (job 020c ran without CLAP) -> d_clap per position for Fig. 3(b). No generation.
set -o pipefail
export EXPC_RES=results/020c_sa3_expC EXPC_RUNS=runs/020c_sa3_expC
python w2s/scripts/rescore_expC_clap.py
