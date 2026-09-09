#!/usr/bin/env bash
# Exp D v2: Uniform vs Top-K(dC) (constant lambda) at K in {5,10,15,25}; CLAP with retry.
set -o pipefail
export EXPC_DIR=results/004_expC_burst
python w2s/scripts/expD_budget.py
