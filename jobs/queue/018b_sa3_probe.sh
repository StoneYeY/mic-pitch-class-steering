#!/usr/bin/env bash
# Job 018b: SA3 pitch-class probe (v2: run through `conda run`, since `conda activate` is unavailable in job shells).
# (1) encode MAESTRO clips with the stable-audio-3-medium-base autoencoder, (2) train the MLSP CNN probe on those
# latents with rectified-flow noise augmentation covering the sampler's noise range, (3) report val F1 per level.
set -o pipefail
PY="conda run -n stablenew --no-capture-output python"
$PY - <<'PYC'
import importlib, subprocess, sys
need = []
for m, pipname in [("librosa","librosa"),("soundfile","soundfile"),("pandas","pandas"),("scipy","scipy"),("pretty_midi","pretty_midi"),("matplotlib","matplotlib")]:
    try: importlib.import_module(m)
    except Exception: need.append(pipname)
print("missing in stablenew:", need or "none", flush=True)
if need: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])
PYC
export SA3_DATA=data/sa3_probe SA3_PROBE="$W2S_RESULTS/sa3_probe_best.pt"
export SA3_MAESTRO="$HOME/Desktop/sa_latent_probe/data/maestro_v3_extract/maestro-v3.0.0" SA3_MAX_FILES=120 SA3_CLIPS_PER_FILE=6
export SA3_T_LEVELS="0,0.1,0.2,0.3,0.4,0.5,0.6"
$PY w2s/scripts/sa3_encode_maestro.py && $PY w2s/scripts/sa3_train_probe.py
