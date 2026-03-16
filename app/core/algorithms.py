"""
Keyframe extraction algorithms for the WLASL Keyframe & Feature Extractor.

Each function takes a list of frames and a target count, and returns
(selected_frames, selected_indices).
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA


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


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    """Z-score normalization that safely handles constant vectors."""
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    mu = values.mean(axis=0, keepdims=True)
    sigma = values.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    return (values - mu) / sigma


def _build_frame_embeddings(frames: list[np.ndarray], emb_dim: int) -> np.ndarray:
    """Build compact appearance embeddings from downsampled grayscale frames."""
    vectors = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        vectors.append(small.astype(np.float32).reshape(-1) / 255.0)

    raw = np.asarray(vectors, dtype=np.float32)
    pca_dim = max(1, min(emb_dim, raw.shape[0], raw.shape[1]))
    return PCA(n_components=pca_dim).fit_transform(raw)


def _compute_elbow_angle(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
) -> float:
    """Compute elbow angle in radians from 2D keypoints."""
    if np.any(np.isnan(shoulder)) or np.any(np.isnan(elbow)) or np.any(np.isnan(wrist)):
        return 0.0

    v1 = shoulder - elbow
    v2 = wrist - elbow
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def _extract_multimodal_descriptors(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract [flow_scalar, pose_features, embedding_pca] with quality signals."""
    import mediapipe as mp
    from .landmark_extractor import _get_hand_model_path, _get_pose_model_path

    n = len(frames)
    if n == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # Flow scalar per frame (mean dense magnitude).
    flow_scalar = np.zeros((n, 1), dtype=np.float32)
    if n > 1:
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        for i in range(n - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i],
                gray_frames[i + 1],
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_scalar[i, 0] = float(np.mean(mag))
        flow_scalar[-1, 0] = flow_scalar[-2, 0]

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

    pose_features = np.zeros((n, 7), dtype=np.float32)
    hand_confidence = np.zeros((n,), dtype=np.float32)
    hand_coverage = np.zeros((n,), dtype=np.float32)

    prev_left_wrist = None
    prev_right_wrist = None
    prev_left_hand_center = None
    prev_right_hand_center = None

    pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
    try:
        for i, frame in enumerate(frames):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            pose_result = pose_landmarker.detect(mp_image)
            hand_result = hand_landmarker.detect(mp_image)

            left_shoulder = np.array([np.nan, np.nan], dtype=np.float32)
            left_elbow = np.array([np.nan, np.nan], dtype=np.float32)
            left_wrist = np.array([np.nan, np.nan], dtype=np.float32)
            right_shoulder = np.array([np.nan, np.nan], dtype=np.float32)
            right_elbow = np.array([np.nan, np.nan], dtype=np.float32)
            right_wrist = np.array([np.nan, np.nan], dtype=np.float32)

            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                pose_lm = pose_result.pose_landmarks[0]
                left_shoulder = np.array([pose_lm[11].x, pose_lm[11].y], dtype=np.float32)
                left_elbow = np.array([pose_lm[13].x, pose_lm[13].y], dtype=np.float32)
                left_wrist = np.array([pose_lm[15].x, pose_lm[15].y], dtype=np.float32)
                right_shoulder = np.array([pose_lm[12].x, pose_lm[12].y], dtype=np.float32)
                right_elbow = np.array([pose_lm[14].x, pose_lm[14].y], dtype=np.float32)
                right_wrist = np.array([pose_lm[16].x, pose_lm[16].y], dtype=np.float32)

            left_angle = _compute_elbow_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = _compute_elbow_angle(right_shoulder, right_elbow, right_wrist)

            left_wrist_speed = 0.0
            right_wrist_speed = 0.0
            if prev_left_wrist is not None and not np.any(np.isnan(left_wrist)):
                left_wrist_speed = float(np.linalg.norm(left_wrist - prev_left_wrist))
            if prev_right_wrist is not None and not np.any(np.isnan(right_wrist)):
                right_wrist_speed = float(np.linalg.norm(right_wrist - prev_right_wrist))
            if not np.any(np.isnan(left_wrist)):
                prev_left_wrist = left_wrist
            if not np.any(np.isnan(right_wrist)):
                prev_right_wrist = right_wrist

            left_hand_speed = 0.0
            right_hand_speed = 0.0
            hand_points_all = []
            confidences = []
            if hand_result.hand_landmarks and hand_result.handedness:
                for hand_landmarks, handedness in zip(
                    hand_result.hand_landmarks,
                    hand_result.handedness,
                ):
                    hand_pts = np.array([[p.x, p.y] for p in hand_landmarks], dtype=np.float32)
                    hand_points_all.append(hand_pts)
                    conf = float(handedness[0].score) if handedness else 0.0
                    confidences.append(conf)
                    center = hand_pts.mean(axis=0)
                    label = handedness[0].category_name if handedness else "Unknown"
                    if label == "Left":
                        if prev_left_hand_center is not None:
                            left_hand_speed = float(
                                np.linalg.norm(center - prev_left_hand_center)
                            )
                        prev_left_hand_center = center
                    elif label == "Right":
                        if prev_right_hand_center is not None:
                            right_hand_speed = float(
                                np.linalg.norm(center - prev_right_hand_center)
                            )
                        prev_right_hand_center = center

            if confidences:
                hand_confidence[i] = float(np.mean(confidences))

            if hand_points_all:
                all_pts = np.concatenate(hand_points_all, axis=0)
                min_xy = np.clip(all_pts.min(axis=0), 0.0, 1.0)
                max_xy = np.clip(all_pts.max(axis=0), 0.0, 1.0)
                wh = np.maximum(max_xy - min_xy, 0.0)
                hand_coverage[i] = float(wh[0] * wh[1])

            hand_distance = 0.0
            if not np.any(np.isnan(left_wrist)) and not np.any(np.isnan(right_wrist)):
                hand_distance = float(np.linalg.norm(left_wrist - right_wrist))

            pose_features[i] = np.array(
                [
                    left_angle,
                    right_angle,
                    left_wrist_speed,
                    right_wrist_speed,
                    left_hand_speed,
                    right_hand_speed,
                    hand_distance,
                ],
                dtype=np.float32,
            )
    finally:
        pose_landmarker.close()
        hand_landmarker.close()

    embedding_pca = _build_frame_embeddings(frames, emb_dim=12)
    descriptor = np.concatenate(
        [
            _safe_zscore(flow_scalar),
            _safe_zscore(pose_features),
            _safe_zscore(embedding_pca),
        ],
        axis=1,
    ).astype(np.float32)

    return descriptor, hand_confidence, hand_coverage


def _cluster_and_pick_medoids(
    reduced: np.ndarray,
    target_count: int,
) -> list[int]:
    """Cluster descriptors and pick per-cluster medoids."""
    n = reduced.shape[0]
    k = max(1, min(target_count, n))

    labels = None
    try:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(reduced)
    except Exception:
        if n >= k and k > 1:
            labels = SpectralClustering(
                n_clusters=k,
                affinity="nearest_neighbors",
                n_neighbors=min(10, n - 1),
                random_state=42,
                assign_labels="kmeans",
            ).fit_predict(reduced)
        else:
            labels = np.zeros((n,), dtype=np.int32)

    medoids = []
    for cluster_id in range(k):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue
        points = reduced[members]
        centroid = points.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(points - centroid, axis=1)
        medoids.append(int(members[int(np.argmin(dists))]))
    return sorted(set(medoids))


def _enforce_temporal_diversity(
    selected: list[int],
    n_frames: int,
    target_count: int,
) -> list[int]:
    """Spread selected indices over time by injecting under-covered intervals."""
    if n_frames == 0:
        return []

    result = sorted(set(int(i) for i in selected if 0 <= i < n_frames))
    if not result:
        return list(np.linspace(0, n_frames - 1, min(target_count, n_frames), dtype=int))

    segments = min(target_count, n_frames)
    if segments <= 1:
        return [result[0]]

    boundaries = np.linspace(0, n_frames, segments + 1, dtype=int)
    for s in range(segments):
        start, end = boundaries[s], max(boundaries[s + 1] - 1, boundaries[s])
        has_member = any(start <= idx <= end for idx in result)
        if not has_member:
            center = (start + end) // 2
            result.append(center)

    result = sorted(set(result))
    if len(result) > target_count:
        anchors = np.linspace(0, n_frames - 1, target_count)
        chosen = []
        for anchor in anchors:
            idx = int(result[int(np.argmin(np.abs(np.asarray(result) - anchor)))])
            chosen.append(idx)
        result = sorted(set(chosen))

    return result


def _apply_endpoint_bias(selected: list[int], n_frames: int) -> list[int]:
    """Ensure early/late endpoints are represented."""
    if n_frames <= 0:
        return []

    if not selected:
        return [0, n_frames - 1] if n_frames > 1 else [0]

    result = sorted(set(selected))
    start_anchor = int(max(0, round(0.05 * (n_frames - 1))))
    end_anchor = int(min(n_frames - 1, round(0.95 * (n_frames - 1))))

    if min(result) > start_anchor:
        result.append(start_anchor)
    if max(result) < end_anchor:
        result.append(end_anchor)

    return sorted(set(result))


def _quality_filter_and_fill(
    selected: list[int],
    descriptor: np.ndarray,
    hand_confidence: np.ndarray,
    hand_coverage: np.ndarray,
    target_count: int,
) -> list[int]:
    """Drop low-quality frames and refill using quality-ranked candidates."""
    n = descriptor.shape[0]
    if n == 0:
        return []

    conf_thr = float(max(0.15, np.quantile(hand_confidence, 0.2)))
    cov_thr = float(max(0.0015, np.quantile(hand_coverage, 0.2)))

    quality = (
        0.50 * _safe_zscore(hand_confidence.reshape(-1, 1)).reshape(-1)
        + 0.20 * _safe_zscore(hand_coverage.reshape(-1, 1)).reshape(-1)
        + 0.30 * _safe_zscore(np.linalg.norm(descriptor, axis=1).reshape(-1, 1)).reshape(-1)
    )

    filtered = [
        idx
        for idx in sorted(set(selected))
        if hand_confidence[idx] >= conf_thr and hand_coverage[idx] >= cov_thr
    ]

    if len(filtered) < target_count:
        candidates = np.argsort(quality)[::-1]
        for idx in candidates:
            i = int(idx)
            if i not in filtered and hand_confidence[i] >= conf_thr and hand_coverage[i] >= cov_thr:
                filtered.append(i)
            if len(filtered) >= target_count:
                break

    if len(filtered) < target_count:
        candidates = np.argsort(quality)[::-1]
        for idx in candidates:
            i = int(idx)
            if i not in filtered:
                filtered.append(i)
            if len(filtered) >= target_count:
                break

    return sorted(set(filtered))[:target_count]


def _finalize_selection(selected: list[int], n_frames: int, target_count: int) -> np.ndarray:
    """Guarantee exact target count while preserving temporal spread and endpoint bias."""
    if n_frames == 0:
        return np.asarray([], dtype=np.int32)

    selected = _apply_endpoint_bias(selected, n_frames)
    selected = _enforce_temporal_diversity(selected, n_frames, target_count)

    if len(selected) < target_count:
        anchors = np.linspace(0, n_frames - 1, target_count)
        for anchor in anchors:
            idx = int(round(float(anchor)))
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= target_count:
                break

    if len(selected) > target_count:
        anchors = np.linspace(0, n_frames - 1, target_count)
        src = np.asarray(sorted(set(selected)), dtype=np.int32)
        picked = []
        for anchor in anchors:
            choice = int(src[int(np.argmin(np.abs(src - anchor)))])
            picked.append(choice)
        selected = picked

    return np.asarray(sorted(set(selected))[:target_count], dtype=np.int32)


def multimodal_fusion_extraction(frames: list, target_count: int) -> tuple[list, list]:
    """Method D: Multimodal fusion with descriptor clustering and quality constraints."""
    if len(frames) <= target_count:
        idxs = list(range(len(frames)))
        return [frames[i] for i in idxs], np.asarray(idxs, dtype=np.int32)

    descriptor, hand_confidence, hand_coverage = _extract_multimodal_descriptors(frames)
    if descriptor.shape[0] == 0:
        idxs = list(np.linspace(0, len(frames) - 1, target_count, dtype=int))
        return [frames[i] for i in idxs], np.asarray(idxs, dtype=np.int32)

    reduce_dim = max(2, min(8, descriptor.shape[0], descriptor.shape[1]))
    reduced = PCA(n_components=reduce_dim, random_state=42).fit_transform(descriptor)

    selected = _cluster_and_pick_medoids(reduced, target_count=target_count)
    selected = _enforce_temporal_diversity(selected, len(frames), target_count)
    selected = _apply_endpoint_bias(selected, len(frames))
    selected = _quality_filter_and_fill(
        selected,
        descriptor,
        hand_confidence,
        hand_coverage,
        target_count,
    )
    selected = _finalize_selection(selected, len(frames), target_count)
    selected_frames = [frames[i] for i in selected]
    return selected_frames, selected


# --- Algorithm registry for clean UI dispatch ---
ALGORITHM_MAP = {
    "Multimodal fusion (recommended final approach)": lambda frames, n: (
        multimodal_fusion_extraction(frames, n)
    ),
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
