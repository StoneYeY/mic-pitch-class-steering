#!/usr/bin/env bash
# Job 0a: environment check. Read-only; prints everything the sprint needs to know.
set -o pipefail
echo "### host"; hostname; uname -a; date -Is
echo; echo "### gpu"; nvidia-smi || echo "nvidia-smi not found"
echo; echo "### disk"; df -h . /tmp "$HOME" 2>/dev/null | sort -u
echo; echo "### cpu/mem"; nproc; free -g | head -2
echo; echo "### shell env"; echo "PATH=$PATH"; echo "CONDA_PREFIX=${CONDA_PREFIX:-}"; echo "VIRTUAL_ENV=${VIRTUAL_ENV:-}"
echo; echo "### conda envs"; (conda env list 2>/dev/null || ls -1 "$HOME"/miniconda3/envs "$HOME"/anaconda3/envs 2>/dev/null || echo "no conda found")
echo; echo "### python"; which -a python python3 2>/dev/null; python -V 2>&1; python3 -V 2>&1
echo; echo "### key packages (current python)"
python - <<'EOF' 2>&1
import importlib
for m in ["torch","torchaudio","stable_audio_tools","diffusers","transformers","librosa","soundfile","numpy","scipy","pandas","laion_clap","fadtk","einops","pretty_midi","matplotlib","tqdm","huggingface_hub"]:
    try:
        mod = importlib.import_module(m); print(f"{m:20s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"{m:20s} MISSING ({type(e).__name__})")
try:
    import torch; print("cuda:", torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    print("vram GB:", round(torch.cuda.get_device_properties(0).total_memory/2**30,1) if torch.cuda.is_available() else "")
except Exception as e: print("torch check failed:", e)
EOF
echo; echo "### huggingface cache (SAO weights?)"
ls -la "${HF_HOME:-$HOME/.cache/huggingface}/hub" 2>/dev/null | grep -i -E "stable-audio|stabilityai|t5|clap" || echo "(no SAO-looking entries in HF cache)"
echo; echo "### repo tree (depth 3, no runs/)"
find . -maxdepth 3 -not -path './.git*' -not -path './runs*' -not -path '*/__pycache__*' | sort | head -200
echo; echo "### python files line counts"
find . -name '*.py' -not -path './.git*' -not -path './runs*' | xargs wc -l 2>/dev/null | sort -n | tail -40
echo; echo "### checkpoints / models / data around here"
find . -maxdepth 4 \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.json' -o -name '*.mid' -o -name '*.wav' \) -not -path './.git*' 2>/dev/null | head -60
echo; echo "### git"; git log --oneline -5; git status --short | head -20
echo; echo "DONE env check"
