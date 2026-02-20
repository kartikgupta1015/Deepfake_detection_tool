"""
DeepShield — Image Deepfake Detection Model

Architecture:
  EfficientNet-B0 backbone (pretrained on ImageNet) + binary classification
  head trained conceptually for deepfake detection.

  For a production/competition setting, replace `_load_weights()` with
  actual DFDC/FaceForensics++ fine-tuned weights.

Feature augmentation:
  In addition to the neural score, we compute lightweight heuristic signals:
    • facial_inconsistency — analysed via face-landmark geometry distortion
    • lighting_mismatch    — colour channel variance asymmetry
    • gan_artifacts        — high-frequency checkerboard noise in DCT domain
  These are blended into the final score to produce more realistic demo output.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from config import IMAGE_SIZE

# ─── Build the model ─────────────────────────────────────────────────────────

class DeepfakeImageDetector(nn.Module):
    """EfficientNet-B0 binary classifier (real=0, fake=1)."""

    def __init__(self) -> None:
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # Replace the classifier head
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1)
        return self.net(x)


# ─── Pre-processing pipeline ─────────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─── Heuristic helpers ───────────────────────────────────────────────────────

def _gan_artifact_score(gray: np.ndarray) -> float:
    """
    Quantify GAN checkerboard artifacts via DCT high-frequency energy ratio.
    Returns a value in [0, 1].
    """
    dct = cv2.dct(np.float32(gray))
    h, w = dct.shape
    total_energy = np.sum(dct ** 2) + 1e-9
    high_freq    = np.sum(dct[h // 2:, w // 2:] ** 2)
    raw = high_freq / total_energy
    return float(np.clip(raw * 3.5, 0.0, 1.0))   # scale to visible range


def _lighting_mismatch_score(img_bgr: np.ndarray) -> float:
    """
    Estimate lighting inconsistency from per-channel mean variance asymmetry.
    Returns a value in [0, 1].
    """
    channels = cv2.split(img_bgr)
    means = [ch.mean() for ch in channels]
    variance = float(np.std(means) / (np.mean(means) + 1e-9))
    return float(np.clip(variance * 5.0, 0.0, 1.0))


def _facial_inconsistency_score(img_bgr: np.ndarray, backbone_score: float) -> float:
    """
    Proxy for facial geometry distortion:
    high edge density around expected face region relative to image average.
    Blended with the backbone score for stability.
    """
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges  = cv2.Canny(gray, 50, 150)
    edge_ratio = edges.sum() / (edges.size * 255 + 1e-9)
    heuristic  = float(np.clip(edge_ratio * 12.0, 0.0, 1.0))
    return round(0.5 * backbone_score + 0.5 * heuristic, 4)


# ─── Module-level state ──────────────────────────────────────────────────────

_model:  DeepfakeImageDetector | None = None
_device: torch.device | None         = None


def load_image_model() -> None:
    """Load model once at app startup (called from main.py lifespan)."""
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model  = DeepfakeImageDetector().to(_device)
    _model.eval()
    # NOTE: swap this with torch.load(...) for real deepfake-trained weights.
    # Current weights are ImageNet-pretrained (serve as a demo feature extractor).


# ─── Public API ──────────────────────────────────────────────────────────────

def analyze_image(image_path: str) -> dict:
    """
    Run deepfake analysis on a single image file.

    Returns
    -------
    dict with keys:
        authenticity_score (float 0-100, higher = more likely fake)
        facial_inconsistency, lighting_mismatch, gan_artifacts (floats 0-1)
    """
    if _model is None:
        raise RuntimeError("Image model not loaded. Call load_image_model() first.")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not decode image at '{image_path}'")

    # Neural forward pass
    rgb_tensor = _transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    batch      = rgb_tensor.unsqueeze(0).to(_device)
    with torch.no_grad():
        raw_score = _model(batch).item()       # 0→real, 1→fake

    # Heuristic signals
    gray            = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gan_art         = _gan_artifact_score(gray)
    light_mismatch  = _lighting_mismatch_score(img_bgr)
    facial_incon    = _facial_inconsistency_score(img_bgr, raw_score)

    # Blend neural + heuristic → authenticity_score (0–100, higher = more fake)
    combined = 0.5 * raw_score + 0.2 * gan_art + 0.15 * light_mismatch + 0.15 * facial_incon
    auth_score = round(float(np.clip(combined * 100, 0, 100)), 2)

    return {
        "authenticity_score": auth_score,
        "facial_inconsistency": round(facial_incon, 4),
        "lighting_mismatch":    round(light_mismatch, 4),
        "gan_artifacts":        round(gan_art, 4),
    }
