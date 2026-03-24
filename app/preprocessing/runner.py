"""
Pipeline orchestrator — coordinates all 9 preprocessing steps.

Usage (single video):
    cfg = PipelineConfig()
    pipeline = PreprocessingPipeline(cfg)
    result = pipeline.process_single_video(
        video_path=Path("dataset/raw_video_data/who/who_63229.mp4"),
        label="who",
        video_name="who_63229.mp4",
    )

Usage (batch):
    summary = pipeline.run()
    report  = pipeline.report()
"""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

from .clahe import apply_clahe
from .config import PipelineConfig
from .frame_extraction import extract_frames, probe_fps
from .quality_checks import generate_dataset_report
from .signer_crop import analyze_signer, crop_frames
from .storage import sample_exists, upsert_metadata
from .temporal_trim import compute_wrist_velocity, detect_idle_boundaries
from .video_normalizer import normalize_video
from .video_writer import resize_with_padding, write_video

log = logging.getLogger(__name__)

_PIPELINE_VERSION = "2.0.0"

StepCallback = Callable[[int, int, str], None]  # (step, total_steps, message)
_TOTAL_STEPS = 9


class PreprocessingPipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None) -> None:
        self.cfg = cfg or PipelineConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_single_video(
        self,
        video_path: Path,
        label: str,
        video_name: str,
        *,
        skip_existing: bool = True,
        step_cb: Optional[StepCallback] = None,
    ) -> Dict:
        """
        Run the full pipeline for one source video.

        Returns a dict with keys: success, label, video_name, error (on failure),
        and all metadata fields on success.
        """
        if skip_existing and sample_exists(self.cfg.output_dir, label, video_name):
            log.info("Skipping existing: %s/%s", label, video_name)
            return {
                "success": True,
                "skipped": True,
                "label": label,
                "video_name": video_name,
            }

        try:
            return self._process(video_path, label, video_name, step_cb)
        except Exception as exc:
            log.exception("Pipeline failed for %s/%s", label, video_name)
            return {
                "success": False,
                "label": label,
                "video_name": video_name,
                "error": str(exc),
            }

    def run(
        self,
        *,
        skip_existing: bool = True,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict:
        """
        Batch-process every video under cfg.raw_video_dir.

        Returns summary: {total, processed, skipped, failed, trimmed_count}.
        """
        raw_dir = self.cfg.dataset_dir
        video_files = sorted(raw_dir.rglob("*.mp4"))
        total = len(video_files)
        processed = skipped = failed = trimmed_count = 0

        for idx, video_path in enumerate(video_files):
            label = video_path.parent.name
            video_name = video_path.name

            if progress_cb:
                progress_cb(idx + 1, total, f"{label}/{video_name}")

            result = self.process_single_video(
                video_path,
                label,
                video_name,
                skip_existing=skip_existing,
            )

            if result.get("skipped"):
                skipped += 1
            elif result.get("success"):
                processed += 1
                if result.get("trim_range"):
                    trimmed_count += 1
            else:
                failed += 1

        return {
            "total": total,
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "trimmed_count": trimmed_count,
        }

    def report(self):
        """Return quality-check SampleReport list for all processed videos."""
        return generate_dataset_report(self.cfg.output_dir)

    # ------------------------------------------------------------------
    # Internal 9-step pipeline
    # ------------------------------------------------------------------

    def _process(
        self,
        video_path: Path,
        label: str,
        video_name: str,
        step_cb: Optional[StepCallback],
    ) -> Dict:
        def _cb(step: int, msg: str) -> None:
            if step_cb:
                step_cb(step, _TOTAL_STEPS, msg)

        stem = Path(video_name).stem

        # Step 1: probe raw FPS
        _cb(1, "Probing source FPS")
        raw_fps = probe_fps(video_path)
        log.debug("Step 1 — raw_fps=%.2f", raw_fps)

        # Step 2: normalize to CFR with ffmpeg, then decode frames
        _cb(2, "Normalizing video and extracting frames")
        with tempfile.TemporaryDirectory(prefix="pose2word_preprocess_") as tmp_dir:
            normalized_path = Path(tmp_dir) / f"{stem}.mp4"
            ok = normalize_video(video_path, normalized_path, self.cfg)
            if not ok:
                raise RuntimeError(
                    "Video normalization failed. Check ffmpeg availability."
                )
            frames = extract_frames(normalized_path)
        original_frame_count = len(frames)
        if not frames:
            raise RuntimeError("No frames extracted from video.")
        log.debug("Step 2 — extracted %d frames", original_frame_count)

        # Step 3: apply CLAHE per frame (full frame, before crop)
        _cb(3, "Applying CLAHE")
        frames = apply_clahe(frames, self.cfg)
        log.debug("Step 3 — CLAHE applied to %d frames", len(frames))

        # Step 4: analyse signer — bbox + wrist positions + hand visibility
        _cb(4, "Analysing signer")
        bbox, wrist_positions, hand_in_frame = analyze_signer(frames, self.cfg)
        log.debug("Step 4 — bbox=%s", bbox)

        # Step 5: crop to signer bounding box
        _cb(5, "Cropping to signer")
        frames = crop_frames(frames, bbox)
        x1, y1, x2, y2 = bbox
        crop_h = y2 - y1
        crop_w = x2 - x1
        log.debug("Step 5 — cropped to %dx%d", crop_w, crop_h)

        # Step 6: resize + pad to output_size × output_size
        _cb(6, "Resizing & padding")
        size = self.cfg.output_size
        frames = [resize_with_padding(f, size) for f in frames]
        log.debug("Step 6 — resized to %dx%d", size, size)

        # Step 7: temporal trim (dual signal: velocity + hand visibility)
        _cb(7, "Temporal trimming")
        wrist_vel = compute_wrist_velocity(wrist_positions)
        trim_start, trim_end = detect_idle_boundaries(
            wrist_vel, hand_in_frame, self.cfg.target_fps, self.cfg
        )
        frames = frames[trim_start : trim_end + 1]
        trimmed_frame_count = len(frames)
        log.debug(
            "Step 7 — trim=[%d, %d], %d frames remain",
            trim_start,
            trim_end,
            trimmed_frame_count,
        )

        # Step 8: write output video
        _cb(8, "Writing output video")
        out_dir = self.cfg.output_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{stem}.mp4"
        success, write_error = write_video(
            frames, output_path, self.cfg.target_fps, self.cfg
        )
        if not success:
            raise RuntimeError(write_error or f"write_video() failed for {output_path}")
        log.debug("Step 8 — written to %s", output_path)

        # Step 9: update global metadata.json
        _cb(9, "Updating metadata")
        meta = {
            "source_video": video_name,
            "label": label,
            "raw_fps": raw_fps,
            "output_fps": self.cfg.target_fps,
            "original_frame_count": original_frame_count,
            "trimmed_frame_count": trimmed_frame_count,
            "trim_range": [trim_start, trim_end],
            "crop_bbox": list(bbox),
            "crop_size": [crop_w, crop_h],
            "output_size": [size, size],
            "pipeline_version": _PIPELINE_VERSION,
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        upsert_metadata(self.cfg.output_dir, label, video_name, meta)
        log.debug("Step 9 — metadata updated")

        return {"success": True, "skipped": False, **meta}
