"""Bridge to the original MLSP code base (config.py, src/, scripts/) living at the repo root.

Everything here is import-only glue so the sprint code can reuse the exact MLSP
pipeline loading, probe checkpoint, melodies, prompts and coherence metric.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402  (MLSP config.py)
from src.probe import create_probe  # noqa: E402

# --------------------------------------------------------------------------- the MLSP evaluation sets
MLSP_MELODIES: dict[str, list[int]] = {           # scripts/03_run_inference.py TEST_MELODIES (F minor!)
    "ascending":   [53, 55, 56, 58, 60, 61, 63, 65],
    "descending":  [65, 63, 61, 60, 58, 56, 55, 53],
    "alternating": [53, 56, 60, 65, 60, 56, 53],
}

MLSP_PROMPTS: list[str] = [                        # scripts/03_run_inference.py TEST_PROMPTS (musicpref, "piano")
    "Fast-paced Western Classical music with lively violin harmony, electric cello, piano and string accompaniments that are colourful, well-layered, dense, rich, full, pleasant, cheerful, merry and elegant.",
    "An alternative/indie song with groovy bass, punchy drums, piano chords, and synth melodies that create an addictive and retro sound.",
    "Classical music piece with a gentle piano tune and a theremin playing the main melody, creating a unique and heart-touching atmosphere suitable for an animation movie/TV series.",
    "Upbeat indie rock instrumental with piano melody, simple percussion, distortion guitar, and bass.",
    "A slow and captivating piano piece that evokes sombre and reflective emotions with simple yet effective motifs.",
    "Dramatic contemporary classical piano piece with accentuated playing style suitable for documentary or mystery/horror video game soundtrack.",
    "A piano arpeggio melody with reverb and environmental sounds, suitable for amateur video intros or outros.",
    "Relaxing instrumental with piano and French horn, featuring an ascending progression and arpeggiated chords.",
    "A melodic and emotional duet with piano, electric guitar, drums, and synthesizer arrangements that could be suitable for children.",
]


# --------------------------------------------------------------------------- model loading
def load_pipeline(device: str | torch.device | None = None, dtype=None):
    """StableAudioPipeline exactly as MLSP loads it (fp16 by default)."""
    from diffusers import StableAudioPipeline
    device = device or config.get_device()
    dtype = dtype or config.DTYPE
    config.setup_memory_efficient()
    pipe = StableAudioPipeline.from_pretrained(config.MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_probe(path: str | Path | None = None, device: str | torch.device | None = None,
               probe_type: str = "cnn") -> torch.nn.Module:
    """The MLSP CNN probe (125k params): (B,64,T) latent -> (B,T,12) logits."""
    device = device or config.get_device()
    path = Path(path) if path else (config.CHECKPOINT_DIR / "probe_best.pt")
    probe = create_probe(probe_type=probe_type, in_channels=config.PROBE_IN_CHANNELS,
                         hidden_channels=config.PROBE_HIDDEN_CHANNELS, out_classes=config.PROBE_OUT_CLASSES)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    probe.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return probe.to(device).float().eval()


def probe_probs(probe: torch.nn.Module, z: torch.Tensor) -> torch.Tensor:
    """(B,64,T) latent -> (B,12,T) sigmoid probabilities (channel-first, as w2s expects)."""
    return torch.sigmoid(probe(z.float())).permute(0, 2, 1)


# --------------------------------------------------------------------------- the MLSP coherence metric
_eval_mod = None


def _load_eval_module():
    global _eval_mod
    if _eval_mod is None:
        spec = importlib.util.spec_from_file_location("mlsp_eval", ROOT / "scripts" / "04_evaluate.py")
        _eval_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_eval_mod)
    return _eval_mod


def mlsp_coherence(audio_path: str | Path, notes: list[int], tempo: float = 120.0,
                   note_duration_beats: float = 1.0) -> dict:
    """scripts/04_evaluate.py::compute_melody_coherence, unchanged (pYIN @ 86 fps, C2-C7,
    voiced-coverage >= 20% guard).  Returns dict(coherence, voiced_coverage, ...)."""
    return _load_eval_module().compute_melody_coherence(str(audio_path), list(notes), tempo=tempo,
                                                        note_duration_beats=note_duration_beats)
