"""
Pipeline runner — orchestrates all preprocessing steps for the full dataset.
"""

import csv
import time
from pathlib import Path

from .config import PipelineConfig
from .video_normalizer import normalize_video, check_ffmpeg
from .frame_extraction import extract_frames, trim_idle_segments
from .signer_crop import compute_signer_bbox, crop_and_resize
from .denoise import denoise_frames
from .keypoint_extraction import extract_keypoints
from .optical_flow import FlowExtractor
from .motion_mask import generate_masks
from .sequence_normalizer import normalize_sequence_length
from .storage import save_sample, sample_exists
from .quality_checks import check_sample, generate_dataset_report


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for the WLASL dataset."""

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self._flow_extractor: FlowExtractor | None = None

    @property
    def flow_extractor(self) -> FlowExtractor:
        if self._flow_extractor is None:
            self._flow_extractor = FlowExtractor(self.cfg)
        return self._flow_extractor

    # ------------------------------------------------------------------
    # Single-video processing
    # ------------------------------------------------------------------

    def process_single_video(
        self,
        video_path: Path,
        label: str,
        video_name: str,
        *,
        skip_existing: bool = True,
        skip_flow: bool = False,
    ) -> dict:
        """
        Run the full pipeline on one video file.

        Returns a dict with 'status' ('ok' | 'skipped' | 'error'),
        optional 'report', and timing info.
        """
        output_dir = self.cfg.output_dir
        if skip_existing and sample_exists(output_dir, label, video_name):
            return {"status": "skipped", "video": video_name}

        t0 = time.perf_counter()
        try:
            return self._process(video_path, label, video_name, skip_flow=skip_flow)
        except Exception as exc:
            return {
                "status": "error",
                "video": video_name,
                "error": str(exc),
                "elapsed": time.perf_counter() - t0,
            }

    def _process(
        self,
        video_path: Path,
        label: str,
        video_name: str,
        *,
        skip_flow: bool,
    ) -> dict:
        t0 = time.perf_counter()

        # 1. Normalize video (ffmpeg re-encode to CFR)
        norm_path = self.cfg.normalized_video_dir / label / video_name
        if not norm_path.exists():
            ok = normalize_video(video_path, norm_path, self.cfg)
            if not ok:
                return {
                    "status": "error",
                    "video": video_name,
                    "error": "ffmpeg failed",
                }

        # 2. Extract frames
        frames = extract_frames(norm_path)
        original_count = len(frames)
        if original_count == 0:
            return {
                "status": "error",
                "video": video_name,
                "error": "no frames decoded",
            }

        # 3. Temporal trimming
        frames, trim_start, trim_end = trim_idle_segments(frames, self.cfg)
        trimmed_count = len(frames)

        # 4. Signer localization & crop
        bbox = compute_signer_bbox(frames, self.cfg)
        frames = crop_and_resize(frames, bbox, self.cfg.crop_short_side)
        crop_h, crop_w = frames[0].shape[:2]

        # 5. Denoising
        frames = denoise_frames(frames, self.cfg)

        # 6. Keypoint extraction
        keypoints, kp_confidence = extract_keypoints(frames, self.cfg)

        # 7. Optical flow
        flow_vectors = None
        flow_magnitudes = None
        if not skip_flow:
            flow_vectors, flow_magnitudes = self.flow_extractor.extract(frames)

        # 8. Motion masks
        masks = generate_masks(keypoints, flow_magnitudes, crop_h, crop_w, self.cfg)

        # 10. Sequence length normalization
        data = normalize_sequence_length(
            frames, keypoints, flow_vectors, flow_magnitudes, masks, self.cfg
        )

        # 11. Save
        metadata = {
            "source_video": video_name,
            "label": label,
            "original_frame_count": original_count,
            "trimmed_frame_count": trimmed_count,
            "trim_range": [trim_start, trim_end],
            "crop_bbox": list(bbox),
            "crop_size": [crop_h, crop_w],
            "keypoint_confidence_mean": round(kp_confidence, 4),
            "target_fps": self.cfg.target_fps,
            "target_sequence_length": self.cfg.target_sequence_length,
            "flow_computed": not skip_flow,
            "pipeline_version": "1.0.0",
        }
        sample_dir = save_sample(self.cfg.output_dir, label, video_name, data, metadata)

        # 12. Quality check
        report = check_sample(sample_dir)

        elapsed = time.perf_counter() - t0
        return {
            "status": "ok",
            "video": video_name,
            "elapsed": round(elapsed, 2),
            "report": {
                "passed": report.passed,
                "warnings": report.warnings,
                "keypoint_confidence": report.keypoint_confidence,
                "flow_magnitude_mean": report.flow_magnitude_mean,
            },
        }

    # ------------------------------------------------------------------
    # Full dataset processing
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        skip_existing: bool = True,
        skip_flow: bool = False,
        progress_cb=None,
    ) -> dict:
        """
        Process every video in labels.csv.

        Args:
            skip_existing: Skip videos that already have output on disk.
            skip_flow: Skip RAFT optical flow (much faster, for testing).
            progress_cb: Optional callback(current, total, video_name, result).

        Returns:
            Summary dict with counts and per-video results.
        """
        if not check_ffmpeg():
            raise RuntimeError(
                "ffmpeg not available. Run 'uv sync' to install the bundled binary."
            )

        rows = self._load_labels()
        total = len(rows)
        results: list[dict] = []

        for idx, row in enumerate(rows):
            video_name = row["video_name"]
            label = row["label"]
            rel_path = row["path"]
            video_path = self.cfg.dataset_dir / rel_path

            if not video_path.exists():
                result = {
                    "status": "error",
                    "video": video_name,
                    "error": "file not found",
                }
            else:
                result = self.process_single_video(
                    video_path,
                    label,
                    video_name,
                    skip_existing=skip_existing,
                    skip_flow=skip_flow,
                )

            results.append(result)
            if progress_cb:
                progress_cb(idx + 1, total, video_name, result)

        ok = sum(1 for r in results if r["status"] == "ok")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        errors = sum(1 for r in results if r["status"] == "error")

        return {
            "total": total,
            "processed": ok,
            "skipped": skipped,
            "errors": errors,
            "results": results,
        }

    def _load_labels(self) -> list[dict]:
        rows = []
        with open(self.cfg.labels_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows

    def report(self) -> dict:
        """Generate a quality report for the entire preprocessed dataset."""
        return generate_dataset_report(self.cfg.output_dir)
