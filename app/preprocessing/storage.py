"""
Step 11 — Caching and storage.

Saves preprocessed per-video data (frames, keypoints, flow, masks,
metadata) as numpy arrays and JSON to a structured directory tree.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def save_sample(
    output_dir: Path,
    label: str,
    video_name: str,
    data: dict,
    metadata: dict,
) -> Path:
    """
    Persist one preprocessed video sample to disk.

    Directory layout:
        output_dir / label / video_stem / {frames.npy, keypoints.npy, ...}

    Args:
        output_dir: Root output directory.
        label: Class label (e.g. "who").
        video_name: Original filename (e.g. "who_63229.mp4").
        data: Dict from sequence_normalizer containing numpy arrays.
        metadata: Dict with processing stats.

    Returns:
        Path to the created sample directory.
    """
    stem = Path(video_name).stem
    sample_dir = output_dir / label / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    np.save(sample_dir / "frames.npy", data["frames"])
    np.save(sample_dir / "keypoints.npy", data["keypoints"])

    if data.get("flow_vectors") is not None:
        np.save(sample_dir / "flow_vectors.npy", data["flow_vectors"])
    if data.get("flow_magnitudes") is not None:
        np.save(sample_dir / "flow_magnitudes.npy", data["flow_magnitudes"])
    if data.get("masks") is not None:
        np.save(sample_dir / "masks.npy", data["masks"])
    if data.get("attention_mask") is not None:
        np.save(sample_dir / "attention_mask.npy", data["attention_mask"])

    metadata["processing_timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return sample_dir


def sample_exists(output_dir: Path, label: str, video_name: str) -> bool:
    """Check if a preprocessed sample already exists on disk."""
    stem = Path(video_name).stem
    meta = output_dir / label / stem / "metadata.json"
    return meta.exists()
