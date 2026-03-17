"""
Landmark Extraction Pipeline
==============================
Ported from asl_landmark_extractor.py CLI tool — exact same algorithms.

Extracts and processes MediaPipe landmarks from pre-extracted keyframe
images for word-level ASL sign language recognition.

Pipeline (based on Relative Quantization approach from BdSL research):

    Step 1 — Landmark Selection       (56 landmarks → 168 features per frame)
    Step 2 — Hand Dominance Correction (flip left-dominant → right-dominant)
    Step 3 — Shoulder Midpoint Calib.  (body-relative coordinates)
    Step 4 — Relative Quantization     (local centers + discretization)
    Step 5 — Feature Scaling ×100      (gradient-friendly magnitudes)
    Step 6 — Sequence Padding          (pad/truncate to fixed length)

Input:
    Directory of keyframe images (frame_0.png, frame_1.png, …)

Output:
    NumPy .npy file of shape (target_len, 168)
"""

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# Type alias for progress callbacks: fn(fraction, message)
ProgressCallback = Optional[Callable[[float, str], None]]


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class LandmarkConfig:
    """Pipeline configuration — all tunable parameters."""

    # Sequence
    target_sequence_length: int = 15
    feature_dim: int = 168          # 56 landmarks × 3 (x, y, z)

    # MediaPipe
    mediapipe_complexity: int = 1   # 0 = fast, 1 = balanced, 2 = accurate

    # RQ quantization levels (x, y, z)
    hand_quant_levels: tuple = (10, 10, 5)
    face_quant_levels: tuple = (5, 5, 3)
    pose_quant_levels: tuple = (10, 10, 5)

    # Post-RQ feature scaling
    scale_factor: float = 100.0


@dataclass
class LandmarkResult:
    """Structured result from a single landmark extraction run."""
    landmarks: np.ndarray        # (target_len, 168)
    original_frame_count: int
    dominant_hand: str           # 'left' or 'right'
    hands_detected_left: int
    hands_detected_right: int
    face_detected: int
    video_stem: str = ""


# ─────────────────────────────────────────────
# Landmark Index Maps
# ─────────────────────────────────────────────

# MediaPipe Holistic pose landmark indices (8 selected)
POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]

# MediaPipe face mesh indices for 6 anchor points
FACE_INDICES = [1, 10, 234, 454, 152, 13]

# Feature vector layout (56 landmarks × 3 = 168):
LH_START   = 0       # left hand start
LH_SIZE    = 63      # 21 landmarks × 3
RH_START   = 63      # right hand start
RH_SIZE    = 63
POSE_START = 126
POSE_SIZE  = 24      # 8 landmarks × 3
FACE_START = 150
FACE_SIZE  = 18      # 6 landmarks × 3

# Pose sub-block offsets (relative to POSE_START):
POSE_SHOULDERS_SIZE = 6     # first 6 features = 2 shoulders
POSE_LIMB_OFFSET    = 6     # elbows start at POSE_START + 6
POSE_LIMB_SIZE      = 18    # elbows + wrists + hips = 6 × 3


# ─────────────────────────────────────────────
# Frame Loading
# ─────────────────────────────────────────────

def _extract_frame_number(filename: str) -> int:
    """Pull the trailing integer from filenames like frame_0.png."""
    match = re.search(r'(\d+)', Path(filename).stem.split("frame")[-1])
    if match:
        return int(match.group(1))
    nums = re.findall(r'\d+', Path(filename).stem)
    return int(nums[-1]) if nums else 0


def load_keyframe_images(frames_dir: str) -> list[np.ndarray]:
    """
    Load keyframe images from a directory.
    Expects files named frame_0.png, frame_1.png, …
    Sorted by frame number in ascending order.
    Returns BGR frames.
    """
    frames_path = Path(frames_dir)

    image_files = sorted(
        list(frames_path.glob("frame_*.png")) +
        list(frames_path.glob("frame_*.jpg")),
        key=lambda p: _extract_frame_number(p.name),
    )

    if not image_files:
        raise FileNotFoundError(
            f"No frame_*.png/jpg images found in: {frames_dir}"
        )

    frames = []
    for img_path in image_files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise ValueError(f"Cannot read image: {img_path}")
        frames.append(bgr)

    return frames


# ─────────────────────────────────────────────
# Step 1 — MediaPipe Extraction + Selection
# ─────────────────────────────────────────────

def extract_landmarks(
    frames: list[np.ndarray],
    config: LandmarkConfig,
) -> np.ndarray:
    """
    Run MediaPipe Holistic on each frame and extract 56 selected
    landmarks into a (K, 168) array.
    """
    from mediapipe.python.solutions.holistic import Holistic as MpHolistic

    K = len(frames)
    landmarks = np.zeros((K, config.feature_dim))

    holistic = MpHolistic(
        static_image_mode=True,
        model_complexity=config.mediapipe_complexity,
        min_detection_confidence=0.5,
    )

    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        vec = np.zeros(config.feature_dim)

        # Left hand (21 landmarks → [0:63])
        if result.left_hand_landmarks:
            for j, lm in enumerate(result.left_hand_landmarks.landmark):
                vec[LH_START + j * 3: LH_START + j * 3 + 3] = [lm.x, lm.y, lm.z]

        # Right hand (21 landmarks → [63:126])
        if result.right_hand_landmarks:
            for j, lm in enumerate(result.right_hand_landmarks.landmark):
                vec[RH_START + j * 3: RH_START + j * 3 + 3] = [lm.x, lm.y, lm.z]

        # Pose (8 selected → [126:150])
        if result.pose_landmarks:
            for j, pose_idx in enumerate(POSE_INDICES):
                lm = result.pose_landmarks.landmark[pose_idx]
                vec[POSE_START + j * 3: POSE_START + j * 3 + 3] = [lm.x, lm.y, lm.z]

        # Face anchors (6 selected → [150:168])
        if result.face_landmarks:
            for j, face_idx in enumerate(FACE_INDICES):
                lm = result.face_landmarks.landmark[face_idx]
                vec[FACE_START + j * 3: FACE_START + j * 3 + 3] = [lm.x, lm.y, lm.z]

        landmarks[i] = vec

    holistic.close()
    return landmarks


# ─────────────────────────────────────────────
# Step 2 — Hand Dominance Correction
# ─────────────────────────────────────────────

def detect_dominant_hand(landmarks: np.ndarray) -> str:
    """Detect dominant hand by non-zero landmark activity. Returns 'right' or 'left'."""
    left_activity  = np.count_nonzero(landmarks[:, LH_START:LH_START + LH_SIZE])
    right_activity = np.count_nonzero(landmarks[:, RH_START:RH_START + RH_SIZE])
    return "left" if left_activity > right_activity else "right"


def correct_hand_dominance(landmarks: np.ndarray) -> tuple[np.ndarray, str]:
    """
    If left-dominant, mirror horizontally so dominant hand is always
    in the right-hand slot [63:126].

    Returns (corrected_landmarks, dominant_hand_str).
    """
    dominant = detect_dominant_hand(landmarks)
    if dominant == "right":
        return landmarks, dominant

    out = landmarks.copy()

    for f in range(len(out)):
        # Hands: swap slots + flip x
        lh = out[f, LH_START:LH_START + LH_SIZE].copy()
        rh = out[f, RH_START:RH_START + RH_SIZE].copy()

        if np.any(lh):
            lh[0::3] = 1.0 - lh[0::3]
        if np.any(rh):
            rh[0::3] = 1.0 - rh[0::3]

        out[f, LH_START:LH_START + LH_SIZE] = rh
        out[f, RH_START:RH_START + RH_SIZE] = lh

        # Pose: swap L↔R pairs + flip x
        pose = out[f, POSE_START:POSE_START + POSE_SIZE].copy()
        if np.any(pose):
            pose[0:3], pose[3:6] = pose[3:6].copy(), pose[0:3].copy()
            pose[6:9], pose[9:12] = pose[9:12].copy(), pose[6:9].copy()
            pose[12:15], pose[15:18] = pose[15:18].copy(), pose[12:15].copy()
            pose[18:21], pose[21:24] = pose[21:24].copy(), pose[18:21].copy()
            pose[0::3] = 1.0 - pose[0::3]
            out[f, POSE_START:POSE_START + POSE_SIZE] = pose

        # Face: swap L↔R cheeks + flip x
        face = out[f, FACE_START:FACE_START + FACE_SIZE].copy()
        if np.any(face):
            face[6:9], face[9:12] = face[9:12].copy(), face[6:9].copy()
            face[0::3] = 1.0 - face[0::3]
            out[f, FACE_START:FACE_START + FACE_SIZE] = face

    return out, dominant


# ─────────────────────────────────────────────
# Step 3 — Shoulder Midpoint Calibration
# ─────────────────────────────────────────────

def calibrate_to_shoulders(landmarks: np.ndarray) -> np.ndarray:
    """
    Translate all coordinates so the first frame's shoulder midpoint
    becomes the origin (0, 0, 0). Body-relative rather than camera-relative.
    """
    out = landmarks.copy()

    left_shoulder  = out[0, POSE_START:POSE_START + 3]
    right_shoulder = out[0, POSE_START + 3:POSE_START + 6]
    midpoint = (left_shoulder + right_shoulder) / 2.0

    if np.allclose(midpoint, 0):
        return out

    n_landmarks = (FACE_START + FACE_SIZE) // 3   # 56
    tile = np.tile(midpoint, n_landmarks)
    for f in range(len(out)):
        out[f] -= tile

    return out


# ─────────────────────────────────────────────
# Step 4 & 5 — Relative Quantization (RQ)
# ─────────────────────────────────────────────

def _quantize_block(
    frame: np.ndarray,
    start: int,
    size: int,
    levels: tuple,
    center: np.ndarray,
) -> None:
    """
    In-place: translate a landmark block to a local center,
    then quantize x/y/z to discrete levels.
    """
    n_landmarks = size // 3
    block = frame[start:start + size]

    # Translate to local center
    block -= np.tile(center, n_landmarks)

    # Quantize each axis independently
    for k, n_levels in enumerate(levels):
        coords = block[k::3]
        if len(coords) == 0:
            continue
        mn, mx = coords.min(), coords.max()
        span = mx - mn
        if span < 1e-8:
            coords[:] = 0
        else:
            normed = (coords - mn) / span
            coords[:] = np.floor(normed * n_levels).clip(0, n_levels - 1)


def relative_quantize(
    landmarks: np.ndarray,
    config: LandmarkConfig,
) -> np.ndarray:
    """
    Relative Quantization (RQ) — the paper's most impactful technique.

    Each landmark group is translated to its local physiological center,
    then quantized to discrete levels.

    Local centers:
        Hands       → own wrist (landmark 0 of each hand)
        Pose limbs  → shoulder midpoint (elbows, wrists, hips)
        Face        → nose tip
        Shoulders   → kept as calibrated global coords (NOT quantized)
    """
    out = landmarks.copy()
    K = len(out)

    for f in range(K):
        frame = out[f]

        if np.allclose(frame, 0):
            continue

        # Left hand → relative to left wrist
        lh_wrist = frame[LH_START:LH_START + 3].copy()
        if np.any(lh_wrist):
            _quantize_block(
                frame, LH_START, LH_SIZE,
                config.hand_quant_levels, lh_wrist,
            )

        # Right hand → relative to right wrist
        rh_wrist = frame[RH_START:RH_START + 3].copy()
        if np.any(rh_wrist):
            _quantize_block(
                frame, RH_START, RH_SIZE,
                config.hand_quant_levels, rh_wrist,
            )

        # Pose limbs → relative to shoulder midpoint
        l_sh = frame[POSE_START:POSE_START + 3]
        r_sh = frame[POSE_START + 3:POSE_START + 6]
        shoulder_mid = (l_sh + r_sh) / 2.0

        if np.any(shoulder_mid):
            limb_start = POSE_START + POSE_LIMB_OFFSET
            _quantize_block(
                frame, limb_start, POSE_LIMB_SIZE,
                config.pose_quant_levels, shoulder_mid,
            )

        # Face → relative to nose tip
        nose = frame[FACE_START:FACE_START + 3].copy()
        if np.any(nose):
            _quantize_block(
                frame, FACE_START, FACE_SIZE,
                config.face_quant_levels, nose,
            )

        out[f] = frame

    return out


# ─────────────────────────────────────────────
# Step 5 — Feature Scaling
# ─────────────────────────────────────────────

def scale_features(
    landmarks: np.ndarray,
    config: LandmarkConfig,
) -> np.ndarray:
    """Scale features by a constant factor (default ×100)."""
    return landmarks * config.scale_factor


# ─────────────────────────────────────────────
# Step 6 — Sequence Padding
# ─────────────────────────────────────────────

def pad_sequence(
    landmarks: np.ndarray,
    config: LandmarkConfig,
) -> np.ndarray:
    """Pad or truncate to fixed target_sequence_length."""
    K = len(landmarks)
    target = config.target_sequence_length

    if K >= target:
        return landmarks[:target]

    pad = np.zeros((target - K, config.feature_dim))
    return np.vstack([landmarks, pad])


# ─────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    frames_dir: str,
    config: Optional[LandmarkConfig] = None,
    progress_cb: ProgressCallback = None,
) -> LandmarkResult:
    """
    Full landmark extraction + processing pipeline.

    Parameters
    ----------
    frames_dir  : directory containing keyframe images
    config      : pipeline parameters (defaults if None)
    progress_cb : optional callback(fraction, message) for UI

    Returns
    -------
    LandmarkResult with processed landmarks and metadata.
    """
    if config is None:
        config = LandmarkConfig()

    def _progress(frac: float, msg: str):
        if progress_cb:
            progress_cb(frac, msg)

    stem = Path(frames_dir).name

    # Stage 1 — Load keyframes
    _progress(0.0, "Loading keyframe images…")
    frames = load_keyframe_images(frames_dir)
    original_len = len(frames)

    # Stage 2 — MediaPipe extraction
    _progress(0.05, f"MediaPipe Holistic on {original_len} frames…")
    landmarks = extract_landmarks(frames, config)

    # Detection stats
    detected_lh = sum(
        1 for k in range(original_len)
        if np.any(landmarks[k, LH_START:LH_START + LH_SIZE])
    )
    detected_rh = sum(
        1 for k in range(original_len)
        if np.any(landmarks[k, RH_START:RH_START + RH_SIZE])
    )
    detected_face = sum(
        1 for k in range(original_len)
        if np.any(landmarks[k, FACE_START:FACE_START + FACE_SIZE])
    )

    # Stage 3 — Hand dominance correction
    _progress(0.50, "Hand dominance correction…")
    landmarks, dominant = correct_hand_dominance(landmarks)

    # Stage 4 — Shoulder midpoint calibration
    _progress(0.60, "Shoulder midpoint calibration…")
    landmarks = calibrate_to_shoulders(landmarks)

    # Stage 5 — Relative Quantization + scaling
    _progress(0.70, "Relative Quantization…")
    landmarks = relative_quantize(landmarks, config)
    landmarks = scale_features(landmarks, config)

    # Stage 6 — Sequence padding
    _progress(0.90, f"Padding to {config.target_sequence_length} frames…")
    landmarks = pad_sequence(landmarks, config)

    _progress(1.0, "Done")

    return LandmarkResult(
        landmarks=landmarks,
        original_frame_count=original_len,
        dominant_hand=dominant,
        hands_detected_left=detected_lh,
        hands_detected_right=detected_rh,
        face_detected=detected_face,
        video_stem=stem,
    )


# ─────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────

def save_landmarks(
    landmarks: np.ndarray,
    output_path: str,
) -> str:
    """Save processed landmarks as .npy file. Returns the saved path."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, landmarks)
    return output_path


# ─────────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────────

def discover_frame_dirs(
    root_dir: str,
) -> list[tuple[str, str, str]]:
    """
    Walk a keyframe directory structured as:
        root_dir / label / video_stem / frame_0.png, …

    Returns list of (frames_dir_path, label, video_stem) tuples.
    """
    root = Path(root_dir)
    dirs = []

    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for video_dir in sorted(label_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            # Check if contains frame images
            has_frames = (
                list(video_dir.glob("frame_*.png")) or
                list(video_dir.glob("frame_*.jpg"))
            )
            if has_frames:
                dirs.append((str(video_dir), label, video_dir.name))

    return dirs


def run_batch(
    root_dir: str,
    output_root: str,
    config: Optional[LandmarkConfig] = None,
    video_cb: Optional[Callable[[int, int, str, str], None]] = None,
) -> list[dict]:
    """
    Process all video keyframe directories in a dataset.

    Input structure:
        root_dir / label / video_stem / frame_0.png, frame_1.png, …

    Output structure:
        output_root / label / video_stem.npy

    Parameters
    ----------
    root_dir    : root directory of keyframe images
    output_root : root output directory for .npy files
    config      : pipeline parameters
    video_cb    : callback(current_idx, total, label, video_stem) per video

    Returns
    -------
    List of result dicts: {label, video_stem, status, shape, error}
    """
    if config is None:
        config = LandmarkConfig()

    frame_dirs = discover_frame_dirs(root_dir)
    if not frame_dirs:
        raise FileNotFoundError(f"No keyframe directories found in: {root_dir}")

    results = []

    for i, (frames_dir, label, video_stem) in enumerate(frame_dirs):
        if video_cb:
            video_cb(i, len(frame_dirs), label, video_stem)

        output_dir = os.path.join(output_root, label)
        output_path = os.path.join(output_dir, f"{video_stem}.npy")

        try:
            result = run_pipeline(frames_dir, config)
            save_landmarks(result.landmarks, output_path)
            results.append({
                "label": label,
                "video_stem": video_stem,
                "status": "✓",
                "shape": f"({result.landmarks.shape[0]}, {result.landmarks.shape[1]})",
                "error": None,
            })
        except Exception as e:
            results.append({
                "label": label,
                "video_stem": video_stem,
                "status": "✗",
                "shape": "",
                "error": str(e),
            })

    return results
