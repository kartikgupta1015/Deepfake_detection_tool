"""
DeepShield — Audio Deepfake Detection Model

Approach:
  • Extract 40-coefficient MFCCs + delta + delta-delta features via librosa
  • Compute pitch series and measure pitch irregularity (std-dev normalised)
  • Run a lightweight 1-D CNN on the MFCC matrix to predict fake probability
  • Blend CNN output with pitch irregularity for final score
"""
from __future__ import annotations

import librosa
import numpy as np
import torch
import torch.nn as nn

# ─── Lightweight 1-D CNN on MFCC ─────────────────────────────────────────────

class AudioAntiSpoofCNN(nn.Module):
    """Small CNN operating on (1, n_mfcc, time) MFCC tensors."""

    def __init__(self, n_mfcc: int = 40) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


# ─── Module-level state ──────────────────────────────────────────────────────

_model:  AudioAntiSpoofCNN | None = None
_device: torch.device | None      = None


def load_audio_model() -> None:
    """Load audio model at startup."""
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model  = AudioAntiSpoofCNN().to(_device)
    _model.eval()
    # NOTE: for production replace with fine-tuned ASVspoof weights


# ─── Feature extraction ──────────────────────────────────────────────────────

def _extract_mfcc_tensor(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    fixed_frames: int = 128,
) -> torch.Tensor:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Pad or crop to fixed_frames
    if mfcc.shape[1] < fixed_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, fixed_frames - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :fixed_frames]
    # Normalise
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)
    tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,n,t)
    return tensor


def _pitch_irregularity(y: np.ndarray, sr: int) -> float:
    """Return normalised pitch irregularity in [0, 1]."""
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=float(librosa.note_to_hz("C2")), fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    voiced = f0[voiced_flag] if voiced_flag is not None else f0
    if voiced is None or len(voiced) < 2:
        return 0.3   # neutral fallback
    irr = float(np.std(voiced) / (np.mean(voiced) + 1e-9))
    return float(np.clip(irr, 0.0, 1.0))


# ─── Public API ──────────────────────────────────────────────────────────────

def analyze_audio(audio_path: str) -> dict:
    """
    Analyse an audio file for voice-spoof / deepfake indicators.

    Returns
    -------
    dict
        authenticity_score (0-100, higher = more likely fake),
        synthetic_probability (0-1),
        pitch_irregularity (0-1)
    """
    if _model is None:
        raise RuntimeError("Audio model not loaded. Call load_audio_model() first.")

    # Load audio (mono, 16 kHz for consistency)
    try:
        y, sr = librosa.load(audio_path, sr=16_000, mono=True, duration=30.0)
    except Exception as exc:
        raise ValueError(f"Could not decode audio file: {exc}") from exc

    if len(y) < 512:
        raise ValueError("Audio clip is too short for analysis.")

    # CNN forward pass
    mfcc_tensor = _extract_mfcc_tensor(y, sr).to(_device)
    with torch.no_grad():
        synth_prob = _model(mfcc_tensor).item()    # 0→real, 1→fake

    pitch_irr = _pitch_irregularity(y, sr)

    # Combined score (0-100)
    combined   = 0.65 * synth_prob + 0.35 * pitch_irr
    auth_score = round(float(np.clip(combined * 100, 0, 100)), 2)

    return {
        "authenticity_score": auth_score,
        "synthetic_probability": round(synth_prob, 4),
        "pitch_irregularity":    round(pitch_irr, 4),
    }
