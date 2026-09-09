"""
MIC Project Configuration
Memory optimization for RTX 5080 (16GB)
"""

import torch
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "latents").mkdir(exist_ok=True)
(OUTPUT_DIR / "generated_audio").mkdir(exist_ok=True)
(OUTPUT_DIR / "evaluation").mkdir(exist_ok=True)

# =============================================================================
# Model Settings
# =============================================================================

MODEL_ID = "stabilityai/stable-audio-open-1.0"
DTYPE = torch.float16  # Use fp16 for 16GB GPU
SAMPLE_RATE = 44100

# =============================================================================
# Data Settings
# =============================================================================

CLIP_DURATION = 10.0  # seconds
AUDIO_CHANNELS = 2    # stereo

# Noise augmentation levels (diffusion-style)
NOISE_SIGMAS = [0.0, 0.1, 0.3, 0.5, 0.7]

# =============================================================================
# Probe Settings
# =============================================================================

PROBE_TYPE = "cnn"  # "linear" or "cnn"
PROBE_IN_CHANNELS = 64
PROBE_HIDDEN_CHANNELS = 128
PROBE_OUT_CLASSES = 12  # pitch classes (can change to 128 for piano roll)
PROBE_KERNEL_SIZE = 5

# Training
PROBE_EPOCHS = 50
PROBE_BATCH_SIZE = 16
PROBE_LEARNING_RATE = 1e-3
PROBE_WEIGHT_DECAY = 1e-5

# =============================================================================
# Inference Settings
# =============================================================================

INFERENCE_STEPS = 50
GUIDANCE_SCALE = 4.0
AUDIO_LENGTH = 5.0  # seconds

# Probe guidance
PROBE_GUIDANCE_SCALE = 0.05  # Start small
PROBE_GUIDANCE_START = 0.7   # Start at 70% of denoising (late steps)
PROBE_GUIDANCE_END = 1.0
PROBE_GUIDANCE_EVERY_N = 2   # Every 2 steps
MAX_UPDATE_RATIO = 0.05      # Max update = 5% of latent RMS

# =============================================================================
# Memory Optimization
# =============================================================================

VAE_ENCODE_BATCH_SIZE = 4
CLEAR_CACHE_EVERY_N = 10
USE_GRADIENT_CHECKPOINTING = False  # Enable if OOM


def setup_memory_efficient():
    """Call at start of script for memory optimization"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
