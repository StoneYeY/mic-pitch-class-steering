#!/usr/bin/env bash
# Job 016: read-only environment check for a Stable Audio 3 replication of Exp A/C.
# Answers: which conda env can run SA3 (stable_audio_tools / stable_audio_3)? are the *-base weights
# reachable (HF token + licence gate)? what are the autoencoder latent dims? where is MAESTRO?
set -o pipefail
echo "### date"; date -Is
echo; echo "### disk"; df -h "$HOME" | tail -1
echo; echo "### conda envs with stable-audio libraries"
for env in mic stableaudio stablenew base; do
  echo "--- env: $env"
  conda run -n "$env" python - <<'PY' 2>&1 | grep -v "^$" | head -12
import importlib, sys
print("python", sys.version.split()[0])
for m in ["torch", "stable_audio_tools", "stable_audio_3", "diffusers", "transformers", "huggingface_hub", "einops", "k_diffusion", "alias_free_torch", "descript_audio_codec"]:
    try:
        mod = importlib.import_module(m); print(f"  {m:22s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {m:22s} MISSING ({type(e).__name__})")
PY
done
echo; echo "### huggingface auth + gated access"
python - <<'PY' 2>&1 | head -40
import os, json, glob
from huggingface_hub import HfApi, whoami
try:
    print("whoami:", whoami().get("name"))
except Exception as e:
    print("whoami failed:", type(e).__name__, str(e)[:120])
api = HfApi()
for rid in ["stabilityai/stable-audio-3-medium-base", "stabilityai/stable-audio-3-small-music-base",
            "stabilityai/stable-audio-3-medium", "stabilityai/stable-audio-3-small-music"]:
    try:
        info = api.model_info(rid, files_metadata=True)
        files = [(s.rfilename, round((s.size or 0)/1e9, 2)) for s in info.siblings]
        print(rid, "ACCESSIBLE gated=", info.gated, "files:", files[:12])
    except Exception as e:
        print(rid, "NOT ACCESSIBLE:", type(e).__name__, str(e)[:160])
print()
print("### cached SA3 snapshots + model_config.json")
hub = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface") + "/hub")
for d in sorted(glob.glob(hub + "/models--stabilityai--stable-audio-3*")):
    print(d)
    for f in glob.glob(d + "/snapshots/*/*"):
        print("   ", os.path.basename(f), round(os.path.getsize(f)/1e9, 2), "GB" if os.path.isfile(f) else "")
    for cfg in glob.glob(d + "/snapshots/*/model_config.json"):
        c = json.load(open(cfg))
        print("    sample_rate", c.get("sample_rate"), "audio_channels", c.get("audio_channels"), "model_type", c.get("model_type"))
        pt = c.get("model", {}).get("pretransform", {})
        print("    pretransform:", json.dumps({k: v for k, v in pt.items() if k != "config"})[:300])
        ptc = pt.get("config", {})
        print("    latent_dim", ptc.get("latent_dim"), "downsampling_ratio", ptc.get("downsampling_ratio"), "io_channels", ptc.get("io_channels"))
        dm = c.get("model", {}).get("diffusion", {})
        print("    diffusion:", json.dumps({k: v for k, v in dm.items() if k in ("type", "diffusion_objective")}), "io_channels", dm.get("config", {}).get("io_channels"))
        print("    conditioning:", [x.get("id") for x in c.get("model", {}).get("conditioning", {}).get("configs", [])])
        print("    sampler hints:", {k: c.get(k) for k in ("sample_size", "min_input_length")})
PY
echo; echo "### MAESTRO / MIC data locations"
find "$HOME" -maxdepth 5 -iname "*maestro*" -type d 2>/dev/null | head -10
ls -la "$HOME/Desktop/MIC" 2>/dev/null | head -20
ls "$HOME/Desktop/MIC/data" 2>/dev/null | head; ls "$HOME/Desktop/MIC/outputs" 2>/dev/null | head
find "$HOME" -maxdepth 6 -name "*.midi" 2>/dev/null | head -3
echo; echo "### amg (arousal project) on this box?"
ls -d "$HOME"/Desktop/ICASSP/amg "$HOME"/amg "$HOME"/Desktop/amg 2>/dev/null
echo "DONE sa3 env check"
