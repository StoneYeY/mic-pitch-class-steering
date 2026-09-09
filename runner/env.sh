# Sourced by runner/runner.sh before every job (GPU machine). Adjust the env name if needed.
for f in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh"; do
  [ -f "$f" ] && { source "$f"; break; }
done
command -v conda >/dev/null 2>&1 && conda activate "${W2S_CONDA_ENV:-mic}" 2>/dev/null
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
