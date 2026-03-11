"""
Step 2 & 3 — Frame extraction and temporal trimming.

Extracts frames from normalized videos and trims idle segments at the
start and end by analyzing a motion-energy curve.
"""

import cv2
import numpy as np
from pathlib import Path

from .config import PipelineConfig


def extract_frames(video_path: Path) -> list[np.ndarray]:
    """
    Decode a video file into a list of RGB uint8 numpy arrays.

    Args:
        video_path: Path to the (normalized) video file.

    Returns:
        List of frames, each shape (H, W, 3) dtype uint8.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def compute_motion_energy(frames: list[np.ndarray]) -> np.ndarray:
    """
    Compute per-frame motion energy as the mean absolute difference
    between consecutive frames (grayscale).

    Returns:
        1-D float array of length len(frames).  The first element is 0.
    """
    energy = np.zeros(len(frames), dtype=np.float64)

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY).astype(np.float64)

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float64)
        diff = np.abs(curr_gray - prev_gray)
        energy[i] = diff.mean()
        prev_gray = curr_gray

    return energy


def trim_idle_segments(
    frames: list[np.ndarray],
    cfg: PipelineConfig,
) -> tuple[list[np.ndarray], int, int]:
    """
    Remove idle (motionless) leading and trailing segments.

    Uses motion energy with a threshold of  mean + factor * std  to find
    the first and last "active" frames, then adds a buffer on each side.

    Args:
        frames: Full list of decoded frames.
        cfg: Pipeline config (trim parameters).

    Returns:
        (trimmed_frames, start_idx, end_idx)  — the trimmed list and the
        original indices that define the kept range.
    """
    if len(frames) < cfg.min_active_frames:
        return frames, 0, len(frames) - 1

    energy = compute_motion_energy(frames)
    threshold = energy.mean() + cfg.trim_threshold_factor * energy.std()

    active = np.where(energy > threshold)[0]

    if len(active) == 0:
        # No significant motion detected — keep everything
        return frames, 0, len(frames) - 1

    first_active = int(active[0])
    last_active = int(active[-1])

    # Add buffer
    start = max(0, first_active - cfg.trim_buffer_frames)
    end = min(len(frames) - 1, last_active + cfg.trim_buffer_frames)

    # Ensure minimum length
    if end - start + 1 < cfg.min_active_frames:
        center = (first_active + last_active) // 2
        half = cfg.min_active_frames // 2
        start = max(0, center - half)
        end = min(len(frames) - 1, start + cfg.min_active_frames - 1)

    trimmed = frames[start : end + 1]
    return trimmed, start, end
