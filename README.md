# MIC - Music Inference Control

Probe-based inference-time control for music generation.

## Quick Start (Linux + RTX 5080)

```bash
# 1. Setup environment
conda create -n mic python=3.10 -y
conda activate mic

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test setup
python test_setup.py

# 5. Prepare data (requires MAESTRO dataset)
python scripts/01_prepare_data.py --maestro_path /path/to/maestro --max_files 50

# 6. Train probe
python scripts/02_train_probe.py --epochs 50

# 7. Run inference
python scripts/03_run_inference.py --compare_methods

# 8. Evaluate
python scripts/04_evaluate.py
```

## Project Structure

```
MIC/
├── config.py                # Configuration
├── requirements.txt         # Dependencies
├── test_setup.py           # Environment test
├── setup_guide.md          # Detailed setup guide
│
├── src/
│   ├── probe.py            # Probe architectures
│   ├── data_prep.py        # Dataset preparation
│   ├── train_probe.py      # Probe training
│   ├── evaluate_probe.py   # Probe evaluation
│   ├── inference.py        # Probe-guided generation
│   └── utils.py            # Utility functions
│
├── scripts/
│   ├── 01_prepare_data.py  # Data preparation script
│   ├── 02_train_probe.py   # Training script
│   ├── 03_run_inference.py # Inference script
│   └── 04_evaluate.py      # Evaluation script
│
├── data/                   # Dataset files
├── checkpoints/            # Model checkpoints
└── outputs/                # Generated audio
```

## Contributions

1. **Interpretable Latent Space**: Probes demonstrate that Stable Audio VAE
   encodes pitch information that can be decoded with simple networks.

2. **Learned Loss for Inference Control**: Probes serve as differentiable
   loss functions for gradient-based latent steering during denoising.

## Hardware Requirements

- GPU: RTX 3080+ (10GB+), RTX 5080 (16GB) recommended
- RAM: 32GB+
- CUDA: 12.1+
