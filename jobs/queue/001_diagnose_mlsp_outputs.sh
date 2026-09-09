#!/usr/bin/env bash
# Job 001: analyse the MLSP paper's generated wavs (if present on this machine).
set -o pipefail
python w2s/scripts/diagnose_mlsp_outputs.py --audio_dir "${MLSP_AUDIO_DIR:-$HOME/Desktop/MIC/outputs/generated_audio}" --out_dir "$W2S_RESULTS"
