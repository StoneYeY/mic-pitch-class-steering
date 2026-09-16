#!/usr/bin/env bash
# Job 018: SA3 pitch-class probe. (1) encode MAESTRO clips with the stable-audio-3-medium-base autoencoder,
# (2) train the MLSP CNN probe on those latents with RF-parameterised noise augmentation, (3) report val F1.
set -o pipefail
conda activate stablenew
python - <<'PY'
import importlib, subprocess, sys
need = []
for m, pipname in [("librosa","librosa"),("soundfile","soundfile"),("pandas","pandas"),("scipy","scipy"),("pretty_midi","pretty_midi"),("matplotlib","matplotlib")]:
    try: importlib.import_module(m)
    except Exception: need.append(pipname)
print("missing in stablenew:", need or "none")
if need: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])
PY
export SA3_DATA=data/sa3_probe SA3_PROBE="$W2S_RESULTS/sa3_probe_best.pt"
export SA3_MAESTRO="$HOME/Desktop/sa_latent_probe/data/maestro_v3_extract/maestro-v3.0.0" SA3_MAX_FILES=120 SA3_CLIPS_PER_FILE=6
python w2s/scripts/sa3_encode_maestro.py && python w2s/scripts/sa3_train_probe.py
