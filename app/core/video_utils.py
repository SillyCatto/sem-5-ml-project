"""
Video I/O utilities for the WLASL Keyframe & Feature Extractor.

Includes:
  - get_frames_from_video  : read frames, optionally resampled to target FPS
  - rescale_frames         : resize all frames to 256×256 or 512×512
  - normalize_contrast     : per-frame pixel contrast normalisation (min-max stretch)
  - apply_clahe            : CLAHE contrast enhancement on L channel (LAB space)
  - preprocess_frames      : convenience wrapper for the full pipeline
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Video reading
# ---------------------------------------------------------------------------


def get_frames_from_video(video_path: str, target_fps: float = 30.0) -> list:
    """
    Reads frames from a video file and returns them as a list of RGB images.

    Args:
        video_path : Path to the video file.
        target_fps : Desired output frame rate. If the video's native FPS is
                     higher than target_fps, frames are skipped so the returned
                     list represents exactly target_fps.  If the native FPS is
                     already ≤ target_fps, every frame is returned unchanged.
                     Pass 0 or None to disable resampling.

    Returns:
        List of RGB numpy arrays (H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    resample = target_fps and target_fps > 0 and native_fps > target_fps
    step = native_fps / target_fps if resample else 1.0

    frames = []
    next_keep = 0.0  # fractional frame index of the next frame to keep
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if resample:
            if idx >= next_keep:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                next_keep += step
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1

    cap.release()
    return frames


def get_video_info(video_path: str) -> dict:
    """Return basic metadata (fps, width, height, frame_count) for a video."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


# ---------------------------------------------------------------------------
# Spatial rescaling
# ---------------------------------------------------------------------------

VALID_SIZES = {
    "256 × 256": (256, 256),
    "512 × 512": (512, 512),
}


def rescale_frames(frames: list, target_size: tuple = (256, 256)) -> list:
    """
    Resize every frame to target_size using INTER_AREA (good for downscaling).

    Args:
        frames      : List of RGB numpy arrays.
        target_size : (width, height) tuple — use one of VALID_SIZES values.

    Returns:
        List of resized RGB numpy arrays.
    """
    w, h = target_size
    return [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames]


# ---------------------------------------------------------------------------
# Contrast normalisation
# ---------------------------------------------------------------------------


def normalize_contrast(frames: list) -> list:
    """
    Per-frame min-max contrast stretch to [0, 255].

    Each frame is independently stretched so that its darkest pixel maps to 0
    and its brightest pixel maps to 255.  This corrects exposure differences
    between clips without altering colour balance.

    Args:
        frames : List of RGB numpy arrays (uint8 or float).

    Returns:
        List of uint8 RGB numpy arrays with stretched contrast.
    """
    result = []
    for frame in frames:
        f = frame.astype(np.float32)
        mn = f.min()
        mx = f.max()
        rng = mx - mn
        if rng < 1e-6:
            result.append(frame.astype(np.uint8))
        else:
            stretched = ((f - mn) / rng * 255.0).clip(0, 255).astype(np.uint8)
            result.append(stretched)
    return result


def apply_clahe(
    frames: list,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
) -> list:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to each frame.

    Works in LAB colour space — only the L (luminance) channel is enhanced,
    so colours are preserved.

    Args:
        frames         : List of RGB numpy arrays.
        clip_limit     : Threshold for contrast limiting (higher = more contrast).
        tile_grid_size : Size of the grid for histogram equalisation.

    Returns:
        List of CLAHE-enhanced RGB numpy arrays.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = []
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        result.append(cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2RGB))
    return result


# ---------------------------------------------------------------------------
# Convenience: apply full preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_frames(
    frames: list,
    rescale_size: tuple | None = None,
    run_contrast_norm: bool = False,
    run_clahe: bool = False,
    clahe_clip: float = 2.0,
) -> list:
    """
    Apply the full preprocessing pipeline in order:
        1. Rescale  (if rescale_size is not None)
        2. Contrast normalisation  (if run_contrast_norm)
        3. CLAHE                   (if run_clahe)

    Returns the preprocessed frame list.
    """
    if rescale_size:
        frames = rescale_frames(frames, rescale_size)
    if run_contrast_norm:
        frames = normalize_contrast(frames)
    if run_clahe:
        frames = apply_clahe(frames, clip_limit=clahe_clip)
    return frames
