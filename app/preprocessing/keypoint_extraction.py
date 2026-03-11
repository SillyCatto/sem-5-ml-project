"""
Step 6 — Keypoint / pose extraction using MediaPipe.

Extracts 258-dimensional feature vectors per frame:
  Pose (33 × 4 = 132) + Left Hand (21 × 3 = 63) + Right Hand (21 × 3 = 63).

Applies temporal Savitzky-Golay smoothing to reduce jitter.
"""

import os
import urllib.request

import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter

from .config import PipelineConfig

NUM_POSE_LANDMARKS = 33
POSE_VALS = 4  # x, y, z, visibility
POSE_DIM = NUM_POSE_LANDMARKS * POSE_VALS  # 132

NUM_HAND_LANDMARKS = 21
HAND_VALS = 3  # x, y, z
HAND_DIM = NUM_HAND_LANDMARKS * HAND_VALS  # 63

TOTAL_DIM = POSE_DIM + 2 * HAND_DIM  # 258

_POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
_HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def _ensure_model(url: str, filename: str) -> str:
    os.makedirs(_MODEL_DIR, exist_ok=True)
    path = os.path.join(_MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename} …")
        urllib.request.urlretrieve(url, path)
    return path


def extract_keypoints(
    frames: list[np.ndarray],
    cfg: PipelineConfig,
) -> tuple[np.ndarray, float]:
    """
    Extract pose + hand keypoints from all frames.

    Args:
        frames: List of RGB uint8 frames (cropped).
        cfg: Pipeline configuration.

    Returns:
        (keypoints, mean_confidence)
        keypoints — shape (T, 258) float32, temporally smoothed.
        mean_confidence — average pose visibility across all frames.
    """
    pose_model = _ensure_model(_POSE_MODEL_URL, "pose_landmarker_lite.task")
    hand_model = _ensure_model(_HAND_MODEL_URL, "hand_landmarker.task")

    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=pose_model),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
    )
    hand_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=hand_model),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
    )

    T = len(frames)
    keypoints = np.zeros((T, TOTAL_DIM), dtype=np.float32)
    confidences: list[float] = []

    with (
        mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose_det,
        mp.tasks.vision.HandLandmarker.create_from_options(hand_options) as hand_det,
    ):
        for t, frame in enumerate(frames):
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # --- Pose ---
            pose_result = pose_det.detect(mp_img)
            if pose_result.pose_landmarks:
                lms = pose_result.pose_landmarks[0]
                vec = []
                vis_sum = 0.0
                for lm in lms:
                    vec.extend([lm.x, lm.y, lm.z, lm.visibility])
                    vis_sum += lm.visibility
                keypoints[t, :POSE_DIM] = vec
                confidences.append(vis_sum / len(lms))

            # --- Hands ---
            hand_result = hand_det.detect(mp_img)
            if hand_result.hand_landmarks:
                for hand_idx, (hand_lms, handedness) in enumerate(
                    zip(hand_result.hand_landmarks, hand_result.handedness)
                ):
                    label = handedness[0].category_name  # "Left" or "Right"
                    vec = []
                    for lm in hand_lms:
                        vec.extend([lm.x, lm.y, lm.z])

                    if label == "Left":
                        keypoints[t, POSE_DIM : POSE_DIM + HAND_DIM] = vec
                    else:
                        keypoints[t, POSE_DIM + HAND_DIM :] = vec

    # --- Temporal smoothing (Savitzky-Golay) ---
    win = cfg.keypoint_smooth_window
    order = cfg.keypoint_smooth_order
    if T >= win:
        for col in range(TOTAL_DIM):
            keypoints[:, col] = savgol_filter(keypoints[:, col], win, order)

    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    return keypoints, mean_conf
