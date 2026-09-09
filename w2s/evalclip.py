"""Evaluate one generated clip with every metric the paper reports.

  coherence_mlsp : scripts/04_evaluate.py metric (pYIN @ 86 fps), needs the wav on disk
  coherence_w2s  : same idea at the latent frame rate (w2s.metrics), used for pseudo-labels too
  chroma_cos     : chroma_cqt vs binary target, target-active frames
  clap           : LAION-CLAP (music) text-audio cosine; the audio embedding is kept for FAD
  logmel_l1      : mean |log-mel(x) - log-mel(ref)| against a reference clip (e.g. the unguided
                   same-seed generation) -> "how much did the intervention change the audio"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import metrics
from .data import Melody
from .mlsp_bridge import mlsp_coherence


def logmel(y: np.ndarray, sr: int, n_mels: int = 64, hop: int = 1024) -> np.ndarray:
    import librosa
    m = librosa.feature.melspectrogram(y=metrics.to_mono(y), sr=sr, n_mels=n_mels, hop_length=hop)
    return np.log(m + 1e-6)


class ClipEvaluator:
    def __init__(self, use_clap: bool = True, device: str | None = None, clap_model: str = "laion/clap-htsat-unfused"):
        self.clap = None
        if use_clap:
            try:
                self.clap = metrics.ClapScorer(clap_model, device=device)
                print(f"[evalclip] CLAP loaded: {clap_model} on {self.clap.device}")
            except Exception as e:  # noqa: BLE001
                print(f"[evalclip] CLAP unavailable ({type(e).__name__}: {e}); clap scores will be NaN")
        self._text_cache: dict[str, np.ndarray] = {}

    def text_emb(self, prompt: str) -> np.ndarray | None:
        if self.clap is None:
            return None
        if prompt not in self._text_cache:
            self._text_cache[prompt] = self.clap.text_embed([prompt])[0]
        return self._text_cache[prompt]

    def evaluate(self, audio: np.ndarray, sr: int, melody: Melody, target_aud: np.ndarray, n_aud: int,
                 prompt: str, wav_path: str | Path, ref_audio: np.ndarray | None = None) -> dict:
        """audio: (samples, channels).  target_aud: (12, n_aud) binary target on audible frames."""
        out: dict = {}
        try:
            c = mlsp_coherence(wav_path, list(melody.notes))
            out["coherence_mlsp"] = c["coherence"]; out["voiced_cov_mlsp"] = c["voiced_coverage"]
        except Exception as e:  # noqa: BLE001
            out["coherence_mlsp"] = np.nan; out["voiced_cov_mlsp"] = np.nan; out["err_mlsp"] = str(e)
        mono = metrics.to_mono(audio.T)
        pc, voiced = metrics.pyin_pitch_classes(mono, sr)
        pc, voiced = pc[:n_aud], voiced[:n_aud]
        out["coherence_w2s"] = metrics.melodic_coherence(pc, voiced, target_aud)
        out["voiced_frac"] = float(voiced.mean()) if len(voiced) else np.nan
        out["chroma_cos"] = metrics.chroma_cosine(mono, sr, target_aud)
        first = melody.notes[0] % 12
        act = target_aud.sum(0) > 0
        v = act & voiced
        out["coherence_firstnote"] = float((pc[v] == first).mean()) if v.any() else np.nan   # "all first-note" check
        if self.clap is not None:
            try:
                ae = self.clap.audio_embed([(mono, sr)])[0]
                out["clap"] = float(self.clap.score(ae[None], self.text_emb(prompt)[None])[0])
                out["_clap_emb"] = ae
            except Exception as e:  # noqa: BLE001
                out["clap"] = np.nan; out["err_clap"] = str(e)
        else:
            out["clap"] = np.nan
        if ref_audio is not None:
            a, b = logmel(audio.T, sr), logmel(ref_audio.T, sr)
            n = min(a.shape[1], b.shape[1])
            out["logmel_l1"] = float(np.abs(a[:, :n] - b[:, :n]).mean())
        out["_pc"] = pc; out["_voiced"] = voiced
        return out
