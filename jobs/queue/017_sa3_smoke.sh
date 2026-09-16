#!/usr/bin/env bash
# Job 017: SA3 smoke test. Finds a conda env with stable_audio_tools, inspects how ~/Desktop/amg drove SA3,
# then loads stable-audio-3-medium-base, encodes a MAESTRO clip, runs a 50-step Euler generation with a
# per-step callback, and reports latent dims / VRAM / timing to results/017_sa3_smoke/smoke.json.
set -o pipefail
echo "### which env has stable_audio_tools?"
FOUND=""
for env in stablenew stableaudio mic base; do
  v=$(conda run -n "$env" python -c "import stable_audio_tools,torch;print(getattr(stable_audio_tools,'__version__','ok'),torch.__version__)" 2>/dev/null)
  if [ -n "$v" ]; then echo "  $env: stable_audio_tools $v"; [ -z "$FOUND" ] && FOUND="$env"; else echo "  $env: no"; fi
  v3=$(conda run -n "$env" python -c "import stable_audio_3;print('stable_audio_3 ok')" 2>/dev/null); [ -n "$v3" ] && echo "  $env: $v3"
done
echo "chosen env: ${FOUND:-none}"
echo; echo "### how did ~/Desktop/amg drive SA3?"
grep -rn --include=*.py -E "get_pretrained_model|generate_diffusion_cond|stable_audio_3|StableAudioModel|sampler_type|bfloat16|float16|steps=" "$HOME/Desktop/amg" 2>/dev/null | grep -v "/.git/" | head -30
ls "$HOME/Desktop/amg" 2>/dev/null | head -30
echo; echo "### ~/Desktop/sa_latent_probe (earlier SA latent probe work?)"
ls -la "$HOME/Desktop/sa_latent_probe" 2>/dev/null | head -30
head -40 "$HOME/Desktop/sa_latent_probe/README.md" 2>/dev/null
find "$HOME/Desktop/sa_latent_probe" -maxdepth 2 -name "*.py" 2>/dev/null | head -20
echo; echo "### MAESTRO wav count"
find "$HOME/Desktop/MIC/data/maestro" -name "*.wav" 2>/dev/null | wc -l
if [ -z "$FOUND" ]; then echo "NO ENV WITH stable_audio_tools — install needed"; exit 2; fi
echo; echo "### smoke test in env $FOUND"
export W2S_RESULTS="results/017_sa3_smoke"
conda run -n "$FOUND" --no-capture-output python w2s/scripts/sa3_smoke.py
