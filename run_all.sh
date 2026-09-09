#!/bin/bash
# MIC Full Pipeline Script
# Run this on Linux with RTX 5080

set -e  # Exit on error

echo "=============================================="
echo "MIC - Full Pipeline"
echo "=============================================="

# Check if MAESTRO path is provided
if [ -z "$1" ]; then
    echo "Usage: ./run_all.sh /path/to/maestro [max_files]"
    echo "Example: ./run_all.sh ~/data/maestro 50"
    exit 1
fi

MAESTRO_PATH=$1
MAX_FILES=${2:-50}

echo "MAESTRO path: $MAESTRO_PATH"
echo "Max files: $MAX_FILES"
echo ""

# Activate conda environment
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate mic
fi

# Step 1: Prepare data
echo ""
echo "=============================================="
echo "Step 1: Preparing dataset..."
echo "=============================================="
python scripts/01_prepare_data.py \
    --maestro_path "$MAESTRO_PATH" \
    --max_files $MAX_FILES

# Step 2: Train probe
echo ""
echo "=============================================="
echo "Step 2: Training probe..."
echo "=============================================="
python scripts/02_train_probe.py \
    --epochs 50 \
    --probe_type cnn

# Step 3: Run inference
echo ""
echo "=============================================="
echo "Step 3: Running inference..."
echo "=============================================="
python scripts/03_run_inference.py \
    --compare_methods

# Step 4: Evaluate
echo ""
echo "=============================================="
echo "Step 4: Evaluating results..."
echo "=============================================="
python scripts/04_evaluate.py

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "=============================================="
echo "Results saved to: outputs/"
echo ""
