#!/usr/bin/env python
"""
Test MIC environment setup.

Run this to verify all dependencies are installed correctly.

Usage:
    python test_setup.py
"""

import sys


def test_imports():
    """Test all required imports."""
    print("Testing imports...")

    # PyTorch
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")

    # Diffusers
    import diffusers
    print(f"  Diffusers: {diffusers.__version__}")

    # Transformers
    import transformers
    print(f"  Transformers: {transformers.__version__}")

    # Audio
    import librosa
    print(f"  Librosa: {librosa.__version__}")

    import soundfile
    print(f"  Soundfile: OK")

    import pretty_midi
    print(f"  PrettyMIDI: OK")

    # Sklearn
    import sklearn
    print(f"  Scikit-learn: {sklearn.__version__}")

    print("\nAll imports successful!")
    return True


def test_stable_audio():
    """Test Stable Audio Open loading."""
    print("\nTesting Stable Audio Open...")

    import torch
    from diffusers import StableAudioPipeline

    # Load with fp16
    print("  Loading model (this may take a minute)...")
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print(f"  Model loaded on GPU")
    else:
        print(f"  Model loaded on CPU (will be slow)")

    # Quick test
    print("  Running quick generation test...")
    audio = pipe(
        prompt="Piano melody",
        num_inference_steps=5,  # Very few steps for testing
        audio_end_in_s=1.0,
    ).audios

    print(f"  Generated audio shape: {audio[0].shape}")

    # Memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    print("\nStable Audio test passed!")
    return True


def test_vae_encoding():
    """Test VAE encoding."""
    print("\nTesting VAE encoding...")

    import torch
    from diffusers import StableAudioPipeline

    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16
    )

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # Create dummy audio (10 seconds, stereo, 44.1kHz)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_audio = torch.randn(1, 2, 44100 * 10, dtype=torch.float32).to(device)

    # Encode
    with torch.no_grad():
        latent = pipe.vae.encode(dummy_audio).latent_dist.sample()

    print(f"  Input audio shape: {dummy_audio.shape}")
    print(f"  Latent shape: {latent.shape}")
    print(f"  Latent dtype: {latent.dtype}")

    # Compute fps
    clip_duration = 10.0
    latent_len = latent.shape[-1]
    fps = latent_len / clip_duration
    print(f"  FPS (for labels): {fps:.2f}")

    print("\nVAE encoding test passed!")
    return True


def test_project_modules():
    """Test project module imports."""
    print("\nTesting project modules...")

    sys.path.insert(0, '.')

    try:
        import config
        print(f"  config: OK")

        from src.probe import LinearProbe, CNNProbe, create_probe
        print(f"  src.probe: OK")

        from src.utils import midi_to_pitch_class, add_diffusion_noise
        print(f"  src.utils: OK")

        from src.data_prep import MaestroDataset
        print(f"  src.data_prep: OK")

        from src.train_probe import train_probe
        print(f"  src.train_probe: OK")

        from src.evaluate_probe import evaluate_probe
        print(f"  src.evaluate_probe: OK")

        from src.inference import ProbeGuidedGenerator
        print(f"  src.inference: OK")

    except ImportError as e:
        print(f"  Import error: {e}")
        return False

    print("\nProject modules test passed!")
    return True


def main():
    print("=" * 60)
    print("MIC ENVIRONMENT TEST")
    print("=" * 60)

    all_passed = True

    # Test imports
    try:
        if not test_imports():
            all_passed = False
    except Exception as e:
        print(f"Import test failed: {e}")
        all_passed = False

    # Test project modules
    try:
        if not test_project_modules():
            all_passed = False
    except Exception as e:
        print(f"Project module test failed: {e}")
        all_passed = False

    # Test Stable Audio (optional, requires model download)
    print("\n" + "-" * 60)
    response = input("Test Stable Audio Open? (requires ~10GB download) [y/N]: ")
    if response.lower() == 'y':
        try:
            if not test_stable_audio():
                all_passed = False
        except Exception as e:
            print(f"Stable Audio test failed: {e}")
            all_passed = False

        try:
            if not test_vae_encoding():
                all_passed = False
        except Exception as e:
            print(f"VAE encoding test failed: {e}")
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("Environment is ready for MIC project.")
    else:
        print("SOME TESTS FAILED!")
        print("Please check the errors above and fix dependencies.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
