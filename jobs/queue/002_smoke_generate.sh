#!/usr/bin/env bash
# Job 002: smoke test of the corrected schedule-driven generator (3 generations).
set -o pipefail
python w2s/scripts/smoke_generate.py
