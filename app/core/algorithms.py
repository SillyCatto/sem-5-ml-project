"""
Keyframe extraction algorithms for the WLASL Keyframe & Feature Extractor.

Each function takes a list of frames and a target count, and returns
(selected_frames, selected_indices).
"""

import cv2
import numpy as np


def optical_flow_farneback_dense_extraction(
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
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_scores.append(float(np.mean(mag)))

    flow_scores.append(0)
    flow_scores = np.array(flow_scores)

    top_indices = np.argsort(flow_scores)[::-1][:target_count]
    top_indices = np.sort(top_indices)
    selected_frames = [frames[i] for i in top_indices]
    return selected_frames, top_indices


def optical_flow_lucas_kanade_sparse_extraction(
    frames: list, target_count: int
) -> tuple[list, list]:
    """Selects frames using Lucas-Kanade sparse optical flow."""
    if len(frames) < 2:
        return frames, list(range(len(frames)))

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flow_scores = []

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Detect initial points to track
    prev_points = cv2.goodFeaturesToTrack(gray_frames[0], mask=None, **feature_params)

    for i in range(len(gray_frames) - 1):
        prev = gray_frames[i]
        nxt = gray_frames[i + 1]

        # Calculate optical flow
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, nxt, prev_points, None, **lk_params
        )

        # Keep only good points
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]

        # Calculate motion magnitude
        motion = np.linalg.norm(good_new - good_old, axis=1)
        flow_scores.append(float(np.mean(motion)))

        # Update points for next iteration
        prev_points = good_new.reshape(-1, 1, 2)

    # Pad the last frame score to match length
    flow_scores.append(0)
    flow_scores = np.array(flow_scores)

    top_indices = np.argsort(flow_scores)[::-1][:target_count]
    top_indices = np.sort(top_indices)
    selected_frames = [frames[i] for i in top_indices]
    return selected_frames, top_indices


def keypoint_skeleton_extraction(frames: list, target_count: int) -> tuple[list, list]:
    """Selects frames based on MediaPipe Pose+Hands keypoint movement.

    Uses the MediaPipe Tasks API (PoseLandmarker + HandLandmarker).
    """
    if len(frames) < 2:
        return frames, list(range(len(frames)))

    import mediapipe as mp
    from .landmark_extractor import _get_pose_model_path, _get_hand_model_path

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
            keypoints.append(
                _extract_landmarks(frame, pose_landmarker, hand_landmarker)
            )
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


# --- Algorithm registry for clean UI dispatch ---
ALGORITHM_MAP = {
    "Optical Flow (Motion Magnitude)": lambda frames, n: (
        optical_flow_farneback_dense_extraction(frames, n)
    ),
    "Keypoint/Skeleton (MediaPipe Pose+Hands)": lambda frames, n: (
        keypoint_skeleton_extraction(frames, n)
    ),
    "Lucas-Kanade Sparse Optical Flow": lambda frames, n: (
        optical_flow_lucas_kanade_sparse_extraction(frames, n)
    ),
}

ALGORITHM_NAMES = list(ALGORITHM_MAP.keys())
