"""
Step 12 — Quality checks and validation.

Per-sample checks that flag problematic preprocessing results, plus
a dataset-wide summary report.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SampleReport:
    """Quality metrics for a single preprocessed video."""

    video_name: str
    label: str
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    keypoint_confidence: float = 0.0
    flow_magnitude_mean: float = 0.0
    trimmed_frames: int = 0
    output_frames: int = 0


def check_sample(sample_dir: Path) -> SampleReport:
    """
    Run quality checks on a single preprocessed sample directory.

    Checks:
      - Keypoint confidence > 0.5
      - Flow magnitude mean > 0.5 px
      - At least 10 non-padded frames
      - No extreme outliers in keypoints (> 5σ from global mean)
    """
    meta_path = sample_dir / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    report = SampleReport(
        video_name=meta.get("source_video", sample_dir.name),
        label=meta.get("label", sample_dir.parent.name),
    )

    # --- Keypoints ---
    kp_path = sample_dir / "keypoints.npy"
    if kp_path.exists():
        kp = np.load(kp_path)
        report.output_frames = kp.shape[0]

        # Mean confidence (visibility values at indices 3, 7, 11, … in pose block)
        vis_indices = list(range(3, 132, 4))
        vis_values = kp[:, vis_indices]
        report.keypoint_confidence = (
            float(vis_values[vis_values > 0].mean()) if (vis_values > 0).any() else 0.0
        )

        if report.keypoint_confidence < 0.5:
            report.warnings.append(
                f"Low keypoint confidence: {report.keypoint_confidence:.2f} (< 0.5)"
            )

        # Outlier check
        col_means = kp.mean(axis=0)
        col_stds = kp.std(axis=0) + 1e-8
        z_scores = np.abs((kp - col_means) / col_stds)
        if z_scores.max() > 5.0:
            report.warnings.append(
                f"Keypoint outlier detected: max z-score = {z_scores.max():.1f}"
            )
    else:
        report.warnings.append("keypoints.npy not found")

    # --- Flow ---
    flow_path = sample_dir / "flow_magnitudes.npy"
    if flow_path.exists():
        flow_mag = np.load(flow_path)
        report.flow_magnitude_mean = float(flow_mag.mean())
        if report.flow_magnitude_mean < 0.5:
            report.warnings.append(
                f"Low flow magnitude: {report.flow_magnitude_mean:.2f} (< 0.5 px)"
            )

    # --- Trimmed length ---
    report.trimmed_frames = meta.get("trimmed_frame_count", 0)
    if report.trimmed_frames < 10:
        report.warnings.append(
            f"Very short after trimming: {report.trimmed_frames} frames"
        )

    report.passed = len(report.warnings) == 0
    return report


def generate_dataset_report(output_dir: Path) -> dict:
    """
    Scan all preprocessed samples and produce a summary report.

    Returns a dict with per-class counts, overall stats, and a list of
    failed samples.
    """
    class_counts: dict[str, int] = {}
    failed: list[dict] = []
    total = 0
    passed = 0

    for class_dir in sorted(output_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("_"):
            continue
        class_name = class_dir.name
        count = 0
        for sample_dir in sorted(class_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            meta_path = sample_dir / "metadata.json"
            if not meta_path.exists():
                continue
            total += 1
            count += 1
            report = check_sample(sample_dir)
            if report.passed:
                passed += 1
            else:
                failed.append(
                    {
                        "video": report.video_name,
                        "label": report.label,
                        "warnings": report.warnings,
                    }
                )
        class_counts[class_name] = count

    return {
        "total_samples": total,
        "passed": passed,
        "failed_count": total - passed,
        "class_counts": class_counts,
        "failed_samples": failed,
    }
