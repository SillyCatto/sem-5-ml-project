"""
Step 10 — Sequence length normalization.

Pads or uniformly subsamples a variable-length sequence to a fixed
target length.  Generates attention masks for padded positions.
"""

import numpy as np

from .config import PipelineConfig


def normalize_sequence_length(
    frames: list[np.ndarray],
    keypoints: np.ndarray,
    flow_vectors: np.ndarray | None,
    flow_magnitudes: np.ndarray | None,
    masks: np.ndarray | None,
    cfg: PipelineConfig,
) -> dict:
    """
    Pad or subsample all per-video arrays to *target_sequence_length*.

    Returns a dict with the normalised arrays and an attention mask.
    """
    T_target = cfg.target_sequence_length
    T_current = len(frames)

    if T_current == T_target:
        indices = np.arange(T_current)
        attn_mask = np.ones(T_target, dtype=np.float32)
    elif T_current > T_target:
        indices = np.linspace(0, T_current - 1, T_target, dtype=int)
        attn_mask = np.ones(T_target, dtype=np.float32)
    else:
        # Pad
        indices = np.arange(T_current)
        attn_mask = np.zeros(T_target, dtype=np.float32)
        attn_mask[:T_current] = 1.0

    # --- Frames ---
    sampled_frames = [frames[i] for i in indices]
    if T_current < T_target:
        last = frames[-1]
        sampled_frames.extend([last] * (T_target - T_current))
    frames_arr = np.stack(sampled_frames, axis=0)  # (T, H, W, 3)

    # --- Keypoints ---
    sampled_kp = keypoints[indices]
    if T_current < T_target:
        pad = np.zeros((T_target - T_current, keypoints.shape[1]), dtype=np.float32)
        sampled_kp = np.concatenate([sampled_kp, pad], axis=0)

    # --- Flow (T-1 length) ---
    sampled_flow = None
    sampled_flow_mag = None
    if flow_vectors is not None:
        flow_T = T_target - 1
        flow_current = flow_vectors.shape[0]
        if flow_current == 0:
            h, w = frames[0].shape[:2]
            sampled_flow = np.zeros((flow_T, h, w, 2), dtype=np.float32)
            sampled_flow_mag = np.zeros((flow_T, h, w), dtype=np.float32)
        elif flow_current >= flow_T:
            f_indices = np.linspace(0, flow_current - 1, flow_T, dtype=int)
            sampled_flow = flow_vectors[f_indices]
            sampled_flow_mag = (
                flow_magnitudes[f_indices] if flow_magnitudes is not None else None
            )
        else:
            pad_shape_v = list(flow_vectors.shape)
            pad_shape_v[0] = flow_T - flow_current
            sampled_flow = np.concatenate(
                [flow_vectors, np.zeros(pad_shape_v, dtype=np.float32)], axis=0
            )
            if flow_magnitudes is not None:
                pad_shape_m = list(flow_magnitudes.shape)
                pad_shape_m[0] = flow_T - flow_current
                sampled_flow_mag = np.concatenate(
                    [flow_magnitudes, np.zeros(pad_shape_m, dtype=np.float32)], axis=0
                )

    # --- Masks ---
    sampled_masks = None
    if masks is not None:
        sampled_masks = masks[indices] if T_current >= T_target else masks
        if sampled_masks.shape[0] < T_target:
            pad_shape = list(sampled_masks.shape)
            pad_shape[0] = T_target - sampled_masks.shape[0]
            sampled_masks = np.concatenate(
                [sampled_masks, np.zeros(pad_shape, dtype=np.uint8)], axis=0
            )

    return {
        "frames": frames_arr,
        "keypoints": sampled_kp,
        "flow_vectors": sampled_flow,
        "flow_magnitudes": sampled_flow_mag,
        "masks": sampled_masks,
        "attention_mask": attn_mask,
    }
