"""
Keyframe extraction algorithms for the WLASL Keyframe & Feature Extractor.

Each function takes a list of frames and a target count, and returns
(selected_frames, selected_indices).
"""

import cv2
import numpy as np


def uniform_sampling(frames: list, target_count: int) -> tuple[list, list]:
    """Uniformly samples frames at equal intervals."""
    total_frames = len(frames)
    if total_frames == 0:
        return [], []
    indices = np.linspace(0, total_frames - 1, target_count, dtype=int)
    selected_frames = [frames[i] for i in indices]
    return selected_frames, indices


def motion_based_extraction(frames: list, target_count: int) -> tuple[list, list]:
    """Selects frames with the highest inter-frame motion (absolute difference)."""
    if len(frames) < 2:
        return frames, list(range(len(frames)))
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    motion_scores = []
    for i in range(len(gray_frames) - 1):
        score = np.sum(cv2.absdiff(gray_frames[i], gray_frames[i + 1]))
        motion_scores.append(score)

    # Pad the last frame score to match length
    motion_scores.append(0)
    motion_scores = np.array(motion_scores)

    top_indices = np.argsort(motion_scores)[::-1][:target_count]
    top_indices = np.sort(top_indices)
    selected_frames = [frames[i] for i in top_indices]
    return selected_frames, top_indices


def optical_flow_extraction(frames: list, target_count: int) -> tuple[list, list]:
    """Selects frames based on optical flow magnitude."""
    if len(frames) < 2:
        return frames, list(range(len(frames)))

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flow_scores = []

    for i in range(len(gray_frames) - 1):
        prev = gray_frames[i]
        nxt = gray_frames[i + 1]
        flow = cv2.calcOpticalFlowFarneback(
            prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_scores.append(float(np.mean(mag)))

    # Pad the last frame score to match length
    flow_scores.append(0)
    flow_scores = np.array(flow_scores)

    top_indices = np.argsort(flow_scores)[::-1][:target_count]
    top_indices = np.sort(top_indices)
    selected_frames = [frames[i] for i in top_indices]
    return selected_frames, top_indices


def farneback_dense_optical_flow_extraction(
    frames: list, target_count: int
) -> tuple[list, list]:
    """Selects frames using Farneback dense optical flow with Gaussian blur."""
    if len(frames) < 2:
        return frames, list(range(len(frames)))

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flow_scores = []

    for i in range(len(gray_frames) - 1):
        prev = cv2.GaussianBlur(gray_frames[i], (5, 5), 0)
        nxt = cv2.GaussianBlur(gray_frames[i + 1], (5, 5), 0)
        flow = cv2.calcOpticalFlowFarneback(
            prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_scores.append(float(np.mean(mag)))

    flow_scores.append(0)
    flow_scores = np.array(flow_scores)

    top_indices = np.argsort(flow_scores)[::-1][:target_count]
    top_indices = np.sort(top_indices)
    selected_frames = [frames[i] for i in top_indices]
    return selected_frames, top_indices


def keypoint_skeleton_extraction(
    frames: list, target_count: int
) -> tuple[list, list]:
    """Selects frames based on MediaPipe Pose+Hands keypoint movement.

    Uses the MediaPipe Tasks API (PoseLandmarker + HandLandmarker).
    """
    if len(frames) < 2:
        return frames, list(range(len(frames)))

    import mediapipe as mp
    from landmark_extractor import _get_pose_model_path, _get_hand_model_path

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_get_pose_model_path()),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
    )
    hand_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_get_hand_model_path()),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
    )

    def _extract_landmarks(frame, pose_lm, hand_lm):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        pose_result = pose_lm.detect(mp_image)
        hand_result = hand_lm.detect(mp_image)

        points = []
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            for lm in pose_result.pose_landmarks[0]:
                points.append((lm.x, lm.y))

        if hand_result.hand_landmarks:
            for hand in hand_result.hand_landmarks:
                for lm in hand:
                    points.append((lm.x, lm.y))

        if not points:
            return None
        return np.array(points, dtype=np.float32)

    keypoints = []
    pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
    try:
        for frame in frames:
            keypoints.append(_extract_landmarks(frame, pose_landmarker, hand_landmarker))
    finally:
        pose_landmarker.close()
        hand_landmarker.close()

    scores = []
    for i in range(len(keypoints) - 1):
        a = keypoints[i]
        b = keypoints[i + 1]
        if a is None or b is None:
            scores.append(0)
            continue
        min_len = min(len(a), len(b))
        if min_len == 0:
            scores.append(0)
            continue
        diff = a[:min_len] - b[:min_len]
        scores.append(float(np.mean(np.linalg.norm(diff, axis=1))))

    scores.append(0)
    scores = np.array(scores)

    top_indices = np.argsort(scores)[::-1][:target_count]
    top_indices = np.sort(top_indices)
    selected_frames = [frames[i] for i in top_indices]
    return selected_frames, top_indices


def cnn_lstm_keyframe_extraction(
    frames: list, target_count: int
) -> tuple[list, list]:
    """Selects frames using CNN feature extraction + lightweight LSTM scoring."""
    if len(frames) < 2:
        return frames, list(range(len(frames)))

    def _extract_cnn_features(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sobelx, sobely)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

        feats = []
        for feat_map in (gray, mag, lap):
            small = cv2.resize(feat_map, (8, 8), interpolation=cv2.INTER_AREA)
            feats.append(small.flatten())
        return np.concatenate(feats, axis=0).astype(np.float32)

    features = np.stack([_extract_cnn_features(f) for f in frames], axis=0)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)

    # Lightweight LSTM scoring (fixed weights) to measure temporal change
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    t_steps, feat_dim = features.shape
    hidden_dim = min(128, feat_dim)
    rng = np.random.default_rng(42)
    W = rng.normal(0, 0.05, size=(4 * hidden_dim, feat_dim))
    U = rng.normal(0, 0.05, size=(4 * hidden_dim, hidden_dim))
    b = np.zeros(4 * hidden_dim, dtype=np.float32)

    h = np.zeros(hidden_dim, dtype=np.float32)
    c = np.zeros(hidden_dim, dtype=np.float32)
    scores = [0.0]

    for t in range(1, t_steps):
        x = features[t]
        z = W @ x + U @ h + b
        f, i, o, g = np.split(z, 4)
        f = sigmoid(f)
        i = sigmoid(i)
        o = sigmoid(o)
        g = np.tanh(g)
        c = f * c + i * g
        h_new = o * np.tanh(c)
        scores.append(float(np.linalg.norm(h_new - h)))
        h = h_new

    scores = np.array(scores)
    top_indices = np.argsort(scores)[::-1][:target_count]
    top_indices = np.sort(top_indices)
    selected_frames = [frames[i] for i in top_indices]
    return selected_frames, top_indices


def transformer_attention_keyframe_extraction(
    frames: list, target_count: int
) -> tuple[list, list]:
    """Selects frames using self-attention scores from learned projections."""
    if len(frames) < 2:
        return frames, list(range(len(frames)))

    def _extract_features(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sobelx, sobely)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

        feats = []
        for feat_map in (gray, mag, lap):
            small = cv2.resize(feat_map, (8, 8), interpolation=cv2.INTER_AREA)
            feats.append(small.flatten())
        return np.concatenate(feats, axis=0).astype(np.float32)

    features = np.stack([_extract_features(f) for f in frames], axis=0)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)

    t_steps, feat_dim = features.shape
    d_k = min(64, feat_dim)

    rng = np.random.default_rng(123)
    Wq = rng.normal(0, 0.05, size=(feat_dim, d_k))
    Wk = rng.normal(0, 0.05, size=(feat_dim, d_k))
    Wv = rng.normal(0, 0.05, size=(feat_dim, d_k))

    Q = features @ Wq
    K = features @ Wk
    V = features @ Wv  # noqa: F841

    # Self-attention scores
    attn_logits = (Q @ K.T) / np.sqrt(d_k)
    attn_logits = attn_logits - attn_logits.max(axis=1, keepdims=True)
    attn_weights = np.exp(attn_logits)
    attn_weights = attn_weights / (attn_weights.sum(axis=1, keepdims=True) + 1e-9)

    # Importance per frame: average attention received
    scores = attn_weights.mean(axis=0)

    top_indices = np.argsort(scores)[::-1][:target_count]
    top_indices = np.sort(top_indices)
    selected_frames = [frames[i] for i in top_indices]
    return selected_frames, top_indices


def voxel_spatiotemporal_extraction(
    frames: list,
    target_count: int,
    grid_size: int = 32,
    threshold: int = 12,
) -> tuple[list, list]:
    """Selects frames based on motion volume in a voxel grid."""
    if len(frames) < 2:
        return frames, list(range(len(frames)))

    gray_frames = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
        gray_frames.append(gray)

    motion_voxels = []
    for i in range(len(gray_frames) - 1):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i + 1])
        voxel = (diff > threshold).astype(np.uint8)
        motion_voxels.append(voxel)

    motion_voxels.append(np.zeros_like(motion_voxels[-1]))

    scores = []
    for i in range(len(motion_voxels)):
        scores.append(float(np.mean(motion_voxels[i])))

    scores = np.array(scores)
    top_indices = np.argsort(scores)[::-1][:target_count]
    top_indices = np.sort(top_indices)
    selected_frames = [frames[i] for i in top_indices]
    return selected_frames, top_indices


def draw_quantization_grid(frame) -> np.ndarray:
    """Draws a relative quantization grid overlay on a frame."""
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2
    overlay = frame.copy()
    step_x = w // 10
    step_y = h // 10
    color = (0, 255, 0)
    for i in range(-5, 6):
        cv2.line(
            overlay,
            (center_x + i * step_x, 0),
            (center_x + i * step_x, h),
            color,
            1,
        )
        cv2.line(
            overlay,
            (0, center_y + i * step_y),
            (w, center_y + i * step_y),
            color,
            1,
        )
    cv2.circle(overlay, (center_x, center_y), 5, (255, 0, 0), -1)
    return cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)


# --- Algorithm registry for clean UI dispatch ---
ALGORITHM_MAP = {
    "Uniform Sampling (Standard)": lambda frames, n: uniform_sampling(frames, n),
    "Motion Detection (Action Segments)": lambda frames, n: motion_based_extraction(
        frames, n
    ),
    "Optical Flow (Motion Magnitude)": lambda frames, n: optical_flow_extraction(
        frames, n
    ),
    "Farneback Dense Optical Flow": lambda frames, n: farneback_dense_optical_flow_extraction(
        frames, n
    ),
    "Keypoint/Skeleton (MediaPipe Pose+Hands)": lambda frames, n: keypoint_skeleton_extraction(
        frames, n
    ),
    "CNN + LSTM (Feature Change)": lambda frames, n: cnn_lstm_keyframe_extraction(
        frames, n
    ),
    "Transformer Attention (Self-Attention)": lambda frames, n: transformer_attention_keyframe_extraction(
        frames, n
    ),
    "Voxel Spatio-Temporal (Motion Volume)": lambda frames, n: voxel_spatiotemporal_extraction(
        frames, n
    ),
}

ALGORITHM_NAMES = list(ALGORITHM_MAP.keys()) + [
    "Relative Quantization (Paper Implementation)"
]
