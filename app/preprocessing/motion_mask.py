"""
Step 8 — Background / motion mask generation.

Combines optical-flow thresholding with keypoint-based elliptical ROIs
to produce per-frame binary masks highlighting the signer.
"""

import cv2
import numpy as np

from .config import PipelineConfig
from .keypoint_extraction import (
    NUM_POSE_LANDMARKS,
    POSE_VALS,
    HAND_DIM,
    POSE_DIM,
    NUM_HAND_LANDMARKS,
    HAND_VALS,
)


def flow_mask(magnitude: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """
    Threshold optical-flow magnitude to obtain a binary motion mask.

    Args:
        magnitude: (H, W) float32 flow magnitude for one frame pair.
        cfg: Pipeline config.

    Returns:
        (H, W) uint8 mask, 255 = moving.
    """
    mean_val = magnitude.mean()
    std_val = magnitude.std()
    thresh = mean_val + cfg.mask_threshold_sigma * std_val
    return (magnitude > thresh).astype(np.uint8) * 255


def keypoint_mask(
    keypoints_row: np.ndarray,
    frame_h: int,
    frame_w: int,
    radius: int = 20,
) -> np.ndarray:
    """
    Draw filled ellipses around detected keypoints.

    Args:
        keypoints_row: 1-D array of 258 values for one frame.
        frame_h, frame_w: frame dimensions.
        radius: ellipse radius in pixels.

    Returns:
        (H, W) uint8 mask, 255 = keypoint region.
    """
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)

    # Pose landmarks (x, y, z, vis)
    for i in range(NUM_POSE_LANDMARKS):
        base = i * POSE_VALS
        x, y, _, vis = keypoints_row[base : base + POSE_VALS]
        if vis < 0.3 or (x == 0.0 and y == 0.0):
            continue
        px = int(x * frame_w)
        py = int(y * frame_h)
        cv2.ellipse(mask, (px, py), (radius, radius), 0, 0, 360, 255, -1)

    # Hands (x, y, z)
    for hand_offset in [POSE_DIM, POSE_DIM + HAND_DIM]:
        for i in range(NUM_HAND_LANDMARKS):
            base = hand_offset + i * HAND_VALS
            x, y, _ = keypoints_row[base : base + HAND_VALS]
            if x == 0.0 and y == 0.0:
                continue
            px = int(x * frame_w)
            py = int(y * frame_h)
            cv2.ellipse(mask, (px, py), (radius, radius), 0, 0, 360, 255, -1)

    return mask


def generate_masks(
    keypoints: np.ndarray,
    flow_magnitudes: np.ndarray | None,
    frame_h: int,
    frame_w: int,
    cfg: PipelineConfig,
) -> np.ndarray:
    """
    Combine flow-based and keypoint-based masks for every frame.

    Args:
        keypoints: (T, 258) float32.
        flow_magnitudes: (T-1, H, W) float32 or None if flow was skipped.
        frame_h, frame_w: spatial dimensions of the cropped frames.
        cfg: Pipeline config.

    Returns:
        (T, H, W) uint8 mask array.
    """
    T = keypoints.shape[0]
    masks = np.zeros((T, frame_h, frame_w), dtype=np.uint8)

    for t in range(T):
        kp_m = keypoint_mask(keypoints[t], frame_h, frame_w)

        if flow_magnitudes is not None and t < flow_magnitudes.shape[0]:
            fl_m = flow_mask(flow_magnitudes[t], cfg)
            # Resize flow mask to frame size if dimensions differ
            if fl_m.shape != (frame_h, frame_w):
                fl_m = cv2.resize(
                    fl_m, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST
                )
            combined = cv2.bitwise_or(kp_m, fl_m)
        else:
            combined = kp_m

        # Morphological closing to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        masks[t] = combined

    return masks
