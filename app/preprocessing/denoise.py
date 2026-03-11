"""
Step 5 — Denoising and color normalization.

Mild non-local-means denoising to reduce compression artifacts,
followed by float32 [0, 1] scaling and optional ImageNet normalization
for pretrained backbones.
"""

import cv2
import numpy as np

from .config import PipelineConfig

# ImageNet statistics (used by RAFT and most pretrained vision models)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denoise_frame(frame: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Apply mild non-local-means denoising to a single RGB uint8 frame."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr, None, cfg.denoise_h, cfg.denoise_h_color, 7, 21
    )
    return denoised


def denoise_frames(frames: list[np.ndarray], cfg: PipelineConfig) -> list[np.ndarray]:
    """Denoise all frames (returns RGB uint8)."""
    out = []
    for f in frames:
        denoised_bgr = denoise_frame(f, cfg)
        out.append(cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB))
    return out


def to_float32(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Convert uint8 [0, 255] → float32 [0.0, 1.0]."""
    return [f.astype(np.float32) / 255.0 for f in frames]


def imagenet_normalize(frame: np.ndarray) -> np.ndarray:
    """Apply ImageNet mean/std normalization to a float32 [0, 1] frame."""
    return (frame - IMAGENET_MEAN) / IMAGENET_STD
