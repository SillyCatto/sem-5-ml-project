"""
Inference helpers for upload-and-predict sign recognition.

The inference path is aligned with preprocessing pipeline v2:
normalize -> extract frames -> CLAHE -> signer crop -> resize/pad -> temporal trim.

After those steps, this module extracts 258-dim pose+hands features per frame:
33 pose landmarks x 4 (x, y, z, visibility) +
21 left-hand x 3 +
21 right-hand x 3.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch

from app.preprocessing.clahe import apply_clahe
from app.preprocessing.config import PipelineConfig
from app.preprocessing.frame_extraction import extract_frames
from app.preprocessing.signer_crop import analyze_signer, crop_frames
from app.preprocessing.temporal_trim import (
    compute_wrist_velocity,
    detect_idle_boundaries,
)
from app.preprocessing.video_normalizer import normalize_video
from app.preprocessing.video_writer import resize_with_padding

from model.sign_classifier import create_model


DEFAULT_CLASSES = [
    "brother",
    "call",
    "drink",
    "go",
    "help",
    "man",
    "mother",
    "no",
    "short",
    "tall",
    "what",
    "who",
    "why",
    "woman",
    "yes",
]


@dataclass
class InferenceResult:
    predicted_label: str
    predicted_index: int
    confidence: float
    top_k: list[dict]
    is_confident: bool
    metadata: dict


DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_FLOW_DIM = 32
KEYPOINT_DIM = 258

_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)


def resolve_class_names(
    checkpoint_config: dict | None = None,
    manual_labels: str | None = None,
) -> list[str]:
    """Resolve class names from manual input, training data, or defaults."""
    if manual_labels:
        labels = [label.strip() for label in manual_labels.split(",") if label.strip()]
        if labels:
            return labels

    checkpoint_config = checkpoint_config or {}
    landmarks_dir = checkpoint_config.get("landmarks_dir")
    if landmarks_dir:
        class_dirs = (
            [path for path in Path(landmarks_dir).iterdir() if path.is_dir()]
            if Path(landmarks_dir).exists()
            else []
        )
        class_names = sorted(path.name for path in class_dirs)
        if class_names:
            return class_names

    num_classes = int(checkpoint_config.get("num_classes", len(DEFAULT_CLASSES)))
    if num_classes == len(DEFAULT_CLASSES):
        return DEFAULT_CLASSES.copy()

    return [f"class_{index}" for index in range(num_classes)]


def load_checkpoint(
    checkpoint_path: str | Path, device: str | None = None
) -> tuple[torch.nn.Module, dict, list[str], str]:
    """Load a saved checkpoint and rebuild the matching model."""
    resolved_device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    config = checkpoint.get("config", {})

    model_type = config.get("model_type", "lstm")
    num_classes = int(config.get("num_classes", len(DEFAULT_CLASSES)))
    model_kwargs = _extract_model_kwargs(model_type, config)

    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
        device=resolved_device,
        **model_kwargs,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = resolve_class_names(config)
    if len(class_names) != num_classes:
        class_names = [f"class_{index}" for index in range(num_classes)]

    return model, config, class_names, resolved_device


def build_inference_features(
    video_path: str | Path,
    cfg: PipelineConfig | None = None,
    target_sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, dict]:
    """Run one uploaded video through the preprocessing path used for inference."""
    cfg = cfg or PipelineConfig()
    video_path = Path(video_path)

    with tempfile.TemporaryDirectory(prefix="pose2word_predict_") as tmp_dir:
        normalized_path = Path(tmp_dir) / video_path.name
        ok = normalize_video(video_path, normalized_path, cfg)
        if not ok:
            raise RuntimeError("Video normalization failed. Check ffmpeg availability.")

        frames = extract_frames(normalized_path)
        original_count = len(frames)
        if original_count == 0:
            raise RuntimeError("No frames could be decoded from the uploaded video.")

        frames = apply_clahe(frames, cfg)

        bbox, wrist_positions, hand_in_frame = analyze_signer(frames, cfg)
        frames = crop_frames(frames, bbox)
        x1, y1, x2, y2 = bbox
        crop_h, crop_w = (y2 - y1), (x2 - x1)

        frames = [resize_with_padding(frame, cfg.output_size) for frame in frames]

        wrist_vel = compute_wrist_velocity(wrist_positions)
        trim_start, trim_end = detect_idle_boundaries(
            wrist_vel,
            hand_in_frame,
            cfg.target_fps,
            cfg,
        )
        frames = frames[trim_start : trim_end + 1]
        trimmed_count = len(frames)
        if trimmed_count == 0:
            raise RuntimeError("No active signing segment remained after trimming.")

        keypoints, pose_visibility_mean = _extract_pose_hands_keypoints(frames)
        keypoints = _pad_or_truncate(keypoints, target_sequence_length)

    metadata = {
        "original_frame_count": original_count,
        "trimmed_frame_count": trimmed_count,
        "trim_range": [int(trim_start), int(trim_end)],
        "crop_bbox": [int(value) for value in bbox],
        "crop_size": [int(crop_h), int(crop_w)],
        "pose_visibility_mean": round(float(pose_visibility_mean), 4),
        "target_sequence_length": int(target_sequence_length),
        "target_fps": int(cfg.target_fps),
        "output_size": [int(cfg.output_size), int(cfg.output_size)],
    }
    return keypoints.astype(np.float32), metadata


def predict_video(
    video_path: str | Path,
    checkpoint_path: str | Path,
    *,
    manual_labels: str | None = None,
    confidence_threshold: float = 0.55,
    cfg: PipelineConfig | None = None,
    device: str | None = None,
    top_k: int = 3,
    target_sequence_length: int | None = None,
) -> InferenceResult:
    """Preprocess a video, run inference, and return ranked predictions."""
    model, checkpoint_config, class_names, resolved_device = load_checkpoint(
        checkpoint_path, device=device
    )
    if manual_labels:
        class_names = resolve_class_names(
            checkpoint_config, manual_labels=manual_labels
        )

    if target_sequence_length is None:
        target_sequence_length = int(
            checkpoint_config.get(
                "target_sequence_length",
                checkpoint_config.get("num_frames", DEFAULT_SEQUENCE_LENGTH),
            )
        )

    keypoints, metadata = build_inference_features(
        video_path,
        cfg=cfg,
        target_sequence_length=target_sequence_length,
    )

    landmarks = torch.from_numpy(keypoints).unsqueeze(0).to(resolved_device)
    flow_dim = int(checkpoint_config.get("flow_dim", DEFAULT_FLOW_DIM))
    flow = torch.zeros(
        (1, max(1, landmarks.shape[1] - 1), flow_dim),
        dtype=torch.float32,
        device=resolved_device,
    )

    with torch.no_grad():
        logits = model(landmarks, flow)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    top_indices = np.argsort(probabilities)[::-1][: max(1, top_k)]
    ranked = [
        {
            "label": class_names[int(index)]
            if int(index) < len(class_names)
            else f"class_{int(index)}",
            "index": int(index),
            "confidence": float(probabilities[int(index)]),
        }
        for index in top_indices
    ]

    best = ranked[0]
    metadata["model_type"] = checkpoint_config.get("model_type", "lstm")
    metadata["device"] = resolved_device

    return InferenceResult(
        predicted_label=best["label"],
        predicted_index=best["index"],
        confidence=best["confidence"],
        top_k=ranked,
        is_confident=best["confidence"] >= confidence_threshold,
        metadata=metadata,
    )


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_model_kwargs(model_type: str, config: dict) -> dict:
    landmark_dim = int(config.get("landmark_dim", KEYPOINT_DIM))
    flow_dim = int(config.get("flow_dim", DEFAULT_FLOW_DIM))

    if model_type == "lstm":
        return {
            "landmark_dim": landmark_dim,
            "flow_dim": flow_dim,
            "hidden_dim": int(config.get("hidden_dim", 256)),
            "num_layers": int(config.get("num_layers", 2)),
            "dropout": float(config.get("dropout", 0.3)),
            "bidirectional": bool(config.get("bidirectional", True)),
        }
    if model_type == "transformer":
        return {
            "landmark_dim": landmark_dim,
            "flow_dim": flow_dim,
            "d_model": int(config.get("d_model", 256)),
            "nhead": int(config.get("nhead", 8)),
            "num_layers": int(config.get("num_layers", 4)),
            "dim_feedforward": int(config.get("dim_feedforward", 1024)),
            "dropout": float(config.get("dropout", 0.1)),
        }
    if model_type == "hybrid":
        return {
            "landmark_dim": landmark_dim,
            "flow_dim": flow_dim,
            "hidden_dim": int(config.get("hidden_dim", 256)),
            "num_lstm_layers": int(config.get("num_lstm_layers", 2)),
            "num_attention_heads": int(config.get("num_attention_heads", 4)),
            "dropout": float(config.get("dropout", 0.3)),
        }
    return {}


def _pad_or_truncate(keypoints: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate keypoint sequence to a fixed temporal length."""
    if target_len <= 0:
        raise ValueError("target sequence length must be > 0")

    current_len = keypoints.shape[0]
    if current_len == target_len:
        return keypoints
    if current_len > target_len:
        return keypoints[:target_len]

    pad = np.zeros((target_len - current_len, KEYPOINT_DIM), dtype=np.float32)
    return np.vstack([keypoints, pad])


def _extract_pose_hands_keypoints(
    frames_rgb: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """
    Extract 258-dim keypoints per frame:
      pose (33 x [x,y,z,visibility]) + left hand (21 x [x,y,z]) + right hand (21 x [x,y,z]).
    """
    pose_model = _ensure_task_model("pose_landmarker_lite.task", _POSE_MODEL_URL)
    hand_model = _ensure_task_model("hand_landmarker.task", _HAND_MODEL_URL)

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

    seq = np.zeros((len(frames_rgb), KEYPOINT_DIM), dtype=np.float32)
    pose_vis_scores: list[float] = []

    with mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose:
        with mp.tasks.vision.HandLandmarker.create_from_options(hand_options) as hands:
            for i, frame_rgb in enumerate(frames_rgb):
                vec = np.zeros(KEYPOINT_DIM, dtype=np.float32)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                pose_result = pose.detect(mp_img)
                if pose_result.pose_landmarks:
                    vis_sum = 0.0
                    for j, lm in enumerate(pose_result.pose_landmarks[0]):
                        visibility = float(getattr(lm, "visibility", 0.0))
                        base = j * 4
                        vec[base : base + 4] = [lm.x, lm.y, lm.z, visibility]
                        vis_sum += visibility
                    pose_vis_scores.append(vis_sum / 33.0)
                else:
                    pose_vis_scores.append(0.0)

                hand_result = hands.detect(mp_img)
                for hand_lms, handedness in zip(
                    hand_result.hand_landmarks,
                    hand_result.handedness,
                ):
                    if not handedness:
                        continue
                    label = handedness[0].category_name.lower()
                    offset = 132 if label == "left" else 195
                    for j, lm in enumerate(hand_lms):
                        base = offset + j * 3
                        vec[base : base + 3] = [lm.x, lm.y, lm.z]

                seq[i] = vec

    pose_visibility_mean = float(np.mean(pose_vis_scores)) if pose_vis_scores else 0.0
    return seq, pose_visibility_mean


def _ensure_task_model(filename: str, model_url: str) -> str:
    """Ensure a MediaPipe .task model exists under app/models and return its path."""
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)
    return model_path
