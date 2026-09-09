# MIC Project Environment Setup Guide

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 (10GB) | RTX 5080 (16GB) |
| RAM | 32GB | 64GB |
| Storage | 50GB SSD | 100GB SSD |
| CUDA | 12.1+ | 12.4+ |

**RTX 5080 (16GB) Verdict**: Sufficient for all tasks with fp16 optimization.

---

## Step 1: System Setup (Ubuntu 22.04/24.04)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y build-essential git curl wget vim htop tmux

# Install audio libraries (required for soundfile/librosa)
sudo apt install -y libsndfile1 ffmpeg libavcodec-extra
```

---

## Step 2: NVIDIA Driver + CUDA (for RTX 5080/Blackwell)

```bash
# Check current driver
nvidia-smi

# If no driver, install latest (for RTX 5080, need 550+)
sudo apt install -y nvidia-driver-560  # or latest available

# Reboot
sudo reboot

# Verify
nvidia-smi
```

### Install CUDA Toolkit 12.4+
```bash
# Download CUDA 12.4 (adjust for your Ubuntu version)
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# Install (skip driver if already installed)
sudo sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

---

## Step 3: Python Environment (Miniconda)

```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Create environment
conda create -n mic python=3.10 -y
conda activate mic
```

---

## Step 4: PyTorch (CUDA 12.4)

```bash
# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

---

## Step 5: Project Dependencies

```bash
# Clone project (if using git)
cd ~/projects
git clone https://github.com/fundwotsai2001/MuseControlLite.git
cd MuseControlLite

# Install core dependencies
pip install diffusers==0.27.0
pip install transformers==4.40.0
pip install accelerate==0.28.0
pip install safetensors

# Audio processing
pip install librosa soundfile pretty_midi scipy

# Evaluation
pip install scikit-learn numpy pandas matplotlib

# Jupyter (optional, for notebooks)
pip install jupyter ipywidgets

# HuggingFace login (for Stable Audio)
pip install huggingface_hub
huggingface-cli login
```

---

## Step 6: Download Models

```bash
# Pre-download Stable Audio Open (avoid timeout during training)
python -c "
from diffusers import StableAudioPipeline
print('Downloading Stable Audio Open...')
pipe = StableAudioPipeline.from_pretrained('stabilityai/stable-audio-open-1.0')
print('Done!')
"
```

---

## Step 7: Download MAESTRO Dataset

```bash
# Create data directory
mkdir -p ~/data/maestro

# Download MAESTRO v3.0.0 (~120GB uncompressed)
cd ~/data/maestro
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
unzip maestro-v3.0.0.zip

# Or download smaller subset (2018 only, ~20GB)
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
```

---

## Step 8: Memory Optimization for 16GB GPU

Create `config.py`:
```python
# config.py - Memory optimization settings

import torch

# Use fp16 for inference
DTYPE = torch.float16

# Batch sizes (adjust based on your GPU memory)
PROBE_TRAINING_BATCH_SIZE = 8   # Can be higher, probe is small
INFERENCE_BATCH_SIZE = 1        # Keep at 1 for 16GB GPU

# VAE encoding settings
VAE_ENCODE_BATCH_SIZE = 4       # Batch size for pre-encoding latents

# Gradient checkpointing (saves memory, slower)
USE_GRADIENT_CHECKPOINTING = False  # Enable if OOM

# Clear cache frequency
CLEAR_CACHE_EVERY_N_BATCHES = 10

def setup_memory_efficient():
    """Call this at the start of your script"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
```

---

## Step 9: Verify Installation

Create `test_setup.py`:
```python
#!/usr/bin/env python
"""Test MIC environment setup"""

import sys

def test_imports():
    print("Testing imports...")

    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    import diffusers
    print(f"  Diffusers: {diffusers.__version__}")

    import transformers
    print(f"  Transformers: {transformers.__version__}")

    import librosa
    print(f"  Librosa: {librosa.__version__}")

    import pretty_midi
    print(f"  PrettyMIDI: OK")

    import soundfile
    print(f"  Soundfile: OK")

    print("\nAll imports successful!")

def test_stable_audio():
    print("\nTesting Stable Audio Open...")

    import torch
    from diffusers import StableAudioPipeline

    # Load with fp16 to save memory
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    print(f"  Model loaded successfully!")

    # Quick generation test
    print("  Running quick generation test...")
    audio = pipe(
        prompt="Piano melody",
        num_inference_steps=10,  # Fewer steps for testing
        audio_end_in_s=1.0,      # Short audio for testing
    ).audios

    print(f"  Generated audio shape: {audio[0].shape}")
    print("\nStable Audio test passed!")

    # Memory usage
    print(f"\nGPU Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def test_vae_encoding():
    print("\nTesting VAE encoding...")

    import torch
    from diffusers import StableAudioPipeline

    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    # Create dummy audio (10 seconds, stereo, 44.1kHz)
    dummy_audio = torch.randn(1, 2, 44100 * 10, dtype=torch.float32).cuda()

    # Encode
    with torch.no_grad():
        latent = pipe.vae.encode(dummy_audio).latent_dist.sample()

    print(f"  Input audio shape: {dummy_audio.shape}")
    print(f"  Latent shape: {latent.shape}")
    print(f"  Latent dtype: {latent.dtype}")

    print("\nVAE encoding test passed!")

if __name__ == "__main__":
    test_imports()

    try:
        test_stable_audio()
    except Exception as e:
        print(f"Stable Audio test failed: {e}")
        sys.exit(1)

    try:
        test_vae_encoding()
    except Exception as e:
        print(f"VAE encoding test failed: {e}")
        sys.exit(1)

    print("\n" + "="*50)
    print("All tests passed! Environment is ready.")
    print("="*50)
```

Run the test:
```bash
python test_setup.py
```

---

## Expected Output

```
Testing imports...
  PyTorch: 2.3.0+cu124
  CUDA available: True
  GPU: NVIDIA GeForce RTX 5080
  Memory: 16.0 GB
  Diffusers: 0.27.0
  Transformers: 4.40.0
  Librosa: 0.10.x
  PrettyMIDI: OK
  Soundfile: OK

All imports successful!

Testing Stable Audio Open...
  Model loaded successfully!
  Running quick generation test...
  Generated audio shape: torch.Size([2, 44100])

Stable Audio test passed!

GPU Memory used: 8.50 GB
GPU Memory cached: 10.20 GB

All tests passed! Environment is ready.
```

---

## Troubleshooting

### Out of Memory (OOM)

```python
# Option 1: Use fp16 everywhere
pipe = StableAudioPipeline.from_pretrained(..., torch_dtype=torch.float16)

# Option 2: Enable CPU offload
pipe.enable_model_cpu_offload()

# Option 3: Clear cache frequently
torch.cuda.empty_cache()

# Option 4: Reduce batch size
batch_size = 1
```

### CUDA Version Mismatch

```bash
# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"

# Should match your nvcc version
nvcc --version
```

### RTX 5080 Not Detected

RTX 5080 (Blackwell) requires:
- NVIDIA Driver 560+
- CUDA 12.4+
- PyTorch 2.3+

---

## Project Structure

```
MIC/
├── setup_guide.md           # This file
├── config.py                # Memory optimization config
├── test_setup.py            # Environment test script
│
├── data/
│   └── maestro/             # MAESTRO dataset
│
├── checkpoints/
│   ├── probe_linear_best.pt
│   ├── probe_cnn_best.pt
│   └── adapter_best.pt
│
├── outputs/
│   ├── latents/             # Pre-encoded VAE latents
│   ├── generated_audio/     # Generated samples
│   └── evaluation/          # Metrics results
│
├── src/
│   ├── data_prep.py         # Dataset preparation
│   ├── probe.py             # Probe architectures
│   ├── train_probe.py       # Probe training
│   ├── inference.py         # Probe-guided generation
│   └── evaluate.py          # Evaluation metrics
│
└── notebooks/               # Optional Jupyter notebooks
    ├── 01_data_prep.ipynb
    ├── 02_train_probe.ipynb
    └── 03_inference.ipynb
```

---

## Quick Start Commands

```bash
# 1. Activate environment
conda activate mic

# 2. Pre-encode MAESTRO latents (do once, saves time later)
python src/data_prep.py --maestro_path ~/data/maestro --output_path data/latents

# 3. Train probe
python src/train_probe.py --probe_type cnn --epochs 50

# 4. Run inference experiment
python src/inference.py --probe_checkpoint checkpoints/probe_cnn_best.pt

# 5. Evaluate results
python src/evaluate.py --output_dir outputs/generated_audio
```

---

## Memory Budget for RTX 5080 (16GB)

| Task | Estimated Usage | Safety Margin |
|------|-----------------|---------------|
| Model loading (fp16) | ~5GB | OK |
| VAE encoding | +2GB | OK |
| Probe + gradient | +1GB | OK |
| Inference buffer | +4GB | OK |
| **Total** | **~12GB** | **4GB free** |

You should be fine with RTX 5080!
