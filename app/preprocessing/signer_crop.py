"""
Step 4 — Signer localization and cropping.

Detects the signer using MediaPipe Pose on sampled frames, computes a
stable bounding box, crops all frames, and resizes to a consistent
short-side resolution.
"""

import cv2
import numpy as np
import mediapipe as mp

from .config import PipelineConfig


# Landmark indices used for bounding-box estimation.
# Shoulders (11, 12), elbows (13, 14), wrists (15, 16), hips (23, 24),
# nose (0) — covers the signing space.
_BBOX_LANDMARK_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24]


def _detect_pose_bbox(
    frame_rgb: np.ndarray,
    pose_detector,
) -> tuple[float, float, float, float] | None:
    """
    Run MediaPipe Pose on a single frame and return a normalised
    bounding box (x_min, y_min, x_max, y_max) in [0, 1] coordinates.

    Returns None if no landmarks are detected.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = pose_detector.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None

    landmarks = result.pose_landmarks[0]  # first person
    xs = [landmarks[i].x for i in _BBOX_LANDMARK_IDS if landmarks[i].visibility > 0.3]
    ys = [landmarks[i].y for i in _BBOX_LANDMARK_IDS if landmarks[i].visibility > 0.3]

    if len(xs) < 3:
        return None

    return (min(xs), min(ys), max(xs), max(ys))


def compute_signer_bbox(
    frames: list[np.ndarray],
    cfg: PipelineConfig,
) -> tuple[int, int, int, int]:
    """
    Estimate a stable signer bounding box by running pose detection on
    every *pose_sample_step*-th frame and taking the median bbox.

    Returns:
        (x1, y1, x2, y2) in pixel coordinates of the original frames,
        already expanded by *crop_expansion*.  Clamped to frame bounds.
    """
    import os
    import urllib.request

    # Download pose model if needed
    model_dir = os.path.join(
        os.path.dirname(__file__), "..", "keyframe_extractor", "models"
    )
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pose_landmarker_lite.task")
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, model_path)

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
    )

    h, w = frames[0].shape[:2]
    bboxes: list[tuple[float, float, float, float]] = []

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        for i in range(0, len(frames), cfg.pose_sample_step):
            bbox = _detect_pose_bbox(frames[i], detector)
            if bbox is not None:
                bboxes.append(bbox)

    if not bboxes:
        # Fallback: centre crop 80 % of the frame
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        return (margin_x, margin_y, w - margin_x, h - margin_y)

    # Median across all detections for robustness
    arr = np.array(bboxes)
    x_min, y_min, x_max, y_max = np.median(arr, axis=0)

    # Expand
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    bw, bh = (x_max - x_min) * cfg.crop_expansion, (y_max - y_min) * cfg.crop_expansion
    x_min = cx - bw / 2
    x_max = cx + bw / 2
    y_min = cy - bh / 2
    y_max = cy + bh / 2

    # Convert to pixel coords and clamp
    x1 = max(0, int(x_min * w))
    y1 = max(0, int(y_min * h))
    x2 = min(w, int(x_max * w))
    y2 = min(h, int(y_max * h))

    return (x1, y1, x2, y2)


def crop_and_resize(
    frames: list[np.ndarray],
    bbox: tuple[int, int, int, int],
    short_side: int,
) -> list[np.ndarray]:
    """
    Crop every frame to *bbox* and resize so the short side equals
    *short_side*, preserving aspect ratio.
    """
    x1, y1, x2, y2 = bbox
    crop_h = y2 - y1
    crop_w = x2 - x1

    if crop_h <= 0 or crop_w <= 0:
        # Degenerate bbox — return originals resized
        return _resize_short_side(frames, short_side)

    cropped = [f[y1:y2, x1:x2] for f in frames]
    return _resize_short_side(cropped, short_side)


def _resize_short_side(frames: list[np.ndarray], short_side: int) -> list[np.ndarray]:
    """Resize all frames so the short side == *short_side*, keeping AR."""
    h, w = frames[0].shape[:2]
    if h <= w:
        new_h = short_side
        new_w = int(w * short_side / h)
    else:
        new_w = short_side
        new_h = int(h * short_side / w)

    return [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
