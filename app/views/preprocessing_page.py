"""
Video Preprocessing Page — integrated with the preprocessing pipeline.

Provides a Streamlit UI for:
  - Configuring all pipeline parameters
  - Processing a single video or the full dataset
  - Viewing progress and quality reports
  - Previewing preprocessed outputs
"""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from app.preprocessing.config import PipelineConfig
from app.preprocessing.video_normalizer import check_ffmpeg, normalize_video
from app.preprocessing.frame_extraction import extract_frames, trim_idle_segments
from app.preprocessing.signer_crop import compute_signer_bbox, crop_and_resize
from app.preprocessing.denoise import denoise_frames
from app.preprocessing.keypoint_extraction import extract_keypoints
from app.preprocessing.optical_flow import FlowExtractor
from app.preprocessing.motion_mask import generate_masks
from app.preprocessing.sequence_normalizer import normalize_sequence_length
from app.preprocessing.storage import save_sample, sample_exists
from app.preprocessing.quality_checks import check_sample, generate_dataset_report


# ── Session state keys ───────────────────────────────────────────────────────
_PREFIX = "pp_"


def _key(name: str) -> str:
    return f"{_PREFIX}{name}"


def _init():
    defaults = {
        _key("mode"): "single",
        _key("dataset_dir"): "dataset/raw_video_data",
        _key("output_dir"): "preprocessed",
        _key("preview_frames"): None,
        _key("preview_keypoints"): None,
        _key("preview_flow_mag"): None,
        _key("preview_masks"): None,
        _key("preview_metadata"): None,
        _key("batch_results"): None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main render ──────────────────────────────────────────────────────────────


def render():
    _init()

    st.caption(
        "Run the full preprocessing pipeline on WLASL sign language videos — "
        "normalize, trim, crop, denoise, extract keypoints & optical flow, and cache results."
    )

    # --- ffmpeg check ---
    if not check_ffmpeg():
        st.error(
            "**ffmpeg is not available.**  "
            "Run `uv sync` to install the bundled ffmpeg binary, "
            "or install ffmpeg manually on your system."
        )
        return

    # --- Mode selector ---
    mode = st.radio(
        "Processing Mode",
        ["🎬 Single Video", "📦 Batch (Full Dataset)", "📊 Quality Report"],
        horizontal=True,
        key=_key("mode_radio"),
    )

    st.divider()

    # --- Shared config sidebar ---
    cfg = _render_config()

    if mode == "🎬 Single Video":
        _render_single_video(cfg)
    elif mode == "📦 Batch (Full Dataset)":
        _render_batch(cfg)
    else:
        _render_report(cfg)


# ── Configuration panel ──────────────────────────────────────────────────────


def _render_config() -> PipelineConfig:
    """Render pipeline configuration controls and return a PipelineConfig."""
    with st.expander("⚙️ Pipeline Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Video Normalization**")
            target_fps = st.number_input("Target FPS", 15, 60, 25, key=_key("fps"))
            crf = st.slider(
                "CRF Quality",
                18,
                28,
                20,
                help="Lower = better quality, larger files",
                key=_key("crf"),
            )

            st.markdown("**Temporal Trimming**")
            trim_factor = st.slider(
                "Motion threshold factor",
                0.1,
                3.0,
                1.0,
                0.1,
                help="mean + factor × std  (higher = stricter, trims more idle frames)",
                key=_key("trim_factor"),
            )
            trim_buffer = st.number_input(
                "Buffer frames",
                0,
                30,
                2,
                key=_key("trim_buf"),
                help="Frames to keep before/after first/last active frame",
            )

        with col2:
            st.markdown("**Signer Crop**")
            crop_exp = st.slider(
                "Bbox expansion",
                1.0,
                2.0,
                1.2,
                0.05,
                help="1.2 = 20% padding around detected body",
                key=_key("crop_exp"),
            )
            crop_short = st.selectbox(
                "Short side (px)",
                [224, 256, 320, 384],
                index=1,
                key=_key("crop_short"),
            )

            st.markdown("**Denoising**")
            denoise_h = st.slider(
                "Denoise strength",
                0,
                15,
                6,
                help="0 = disabled, 6 = mild, 10+ = aggressive",
                key=_key("denoise_h"),
            )

        with col3:
            st.markdown("**Optical Flow (RAFT)**")
            flow_model = st.selectbox(
                "RAFT model", ["small", "large"], key=_key("flow_model")
            )
            flow_batch = st.number_input(
                "Flow batch size", 1, 16, 4, key=_key("flow_batch")
            )

            st.markdown("**Sequence**")
            seq_len = st.number_input(
                "Target sequence length", 8, 128, 32, key=_key("seq_len")
            )

            st.markdown("**Device**")
            device = st.selectbox(
                "Compute device",
                ["auto", "cuda", "mps", "cpu"],
                key=_key("device"),
            )

        dataset_dir = st.text_input(
            "Dataset directory", "dataset/raw_video_data", key=_key("ds_dir")
        )
        output_dir = st.text_input(
            "Output directory", "preprocessed", key=_key("out_dir")
        )

    cfg = PipelineConfig(
        dataset_dir=Path(dataset_dir),
        labels_csv=Path(dataset_dir) / "labels.csv",
        output_dir=Path(output_dir),
        normalized_video_dir=Path(output_dir) / "_normalized_videos",
        target_fps=target_fps,
        crf_quality=crf,
        trim_threshold_factor=trim_factor,
        trim_buffer_frames=trim_buffer,
        crop_expansion=crop_exp,
        crop_short_side=crop_short,
        denoise_h=denoise_h,
        denoise_h_color=denoise_h,
        flow_model=flow_model,
        flow_batch_size=flow_batch,
        target_sequence_length=seq_len,
        device=device,
    )
    return cfg


# ── Single video mode ────────────────────────────────────────────────────────


def _render_single_video(cfg: PipelineConfig):
    st.subheader("🎬 Process a Single Video")

    input_method = st.radio(
        "Video source",
        ["Upload file", "Select from dataset"],
        horizontal=True,
        key=_key("input_method"),
    )

    video_path: Path | None = None
    label = ""
    video_name = ""

    if input_method == "Upload file":
        uploaded = st.file_uploader(
            "Upload a sign language video",
            type=["mp4", "mov", "avi", "mkv"],
            key=_key("upload"),
        )
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            tmp.close()
            video_path = Path(tmp.name)
            video_name = uploaded.name
            label = st.text_input(
                "Label (class name)", value="unknown", key=_key("label")
            )
    else:
        # Pick from the dataset's labels.csv
        labels_path = cfg.labels_csv
        if not labels_path.exists():
            st.warning(f"labels.csv not found at `{labels_path}`")
            return

        rows = _read_labels(labels_path)
        if not rows:
            st.warning("labels.csv is empty.")
            return

        options = [f"{r['label']}/{r['video_name']}" for r in rows]
        choice = st.selectbox("Select video", options, key=_key("video_sel"))
        idx = options.index(choice)
        row = rows[idx]
        video_path = cfg.dataset_dir / row["path"]
        label = row["label"]
        video_name = row["video_name"]

        if video_path is None or not video_path.exists():
            st.error(f"File not found: `{video_path}`")
            return

        st.success(f"Selected: **{label}/{video_name}**")

    if video_path is None:
        return

    # --- Skip flow toggle ---
    col_a, col_b = st.columns(2)
    with col_a:
        skip_flow = st.checkbox(
            "Skip optical flow (faster)",
            value=False,
            key=_key("skip_flow"),
        )
    with col_b:
        skip_existing = st.checkbox(
            "Skip if already processed",
            value=True,
            key=_key("skip_existing"),
        )

    # --- Process button ---
    if st.button(
        "▶️ Run Preprocessing Pipeline", type="primary", key=_key("run_single")
    ):
        _run_single_pipeline(
            cfg, video_path, label, video_name, skip_flow, skip_existing
        )

    # --- Preview section ---
    if st.session_state[_key("preview_frames")] is not None:
        _render_preview()


def _run_single_pipeline(
    cfg: PipelineConfig,
    video_path: Path,
    label: str,
    video_name: str,
    skip_flow: bool,
    skip_existing: bool,
):
    """Execute the pipeline with step-by-step status updates."""

    if skip_existing and sample_exists(cfg.output_dir, label, video_name):
        st.info(
            f"⊘ Already processed: `{label}/{video_name}`. "
            "Uncheck 'Skip if already processed' to reprocess."
        )
        _load_preview_from_disk(cfg, label, video_name)
        return

    progress = st.progress(0.0, text="Starting pipeline…")
    status = st.empty()

    try:
        # Step 1: Normalize
        status.info("Step 1/8 — Normalizing video (ffmpeg → CFR)…")
        progress.progress(0.05, text="Normalizing video…")
        norm_path = cfg.normalized_video_dir / label / video_name
        if not norm_path.exists():
            ok = normalize_video(video_path, norm_path, cfg)
            if not ok:
                st.error("❌ ffmpeg normalization failed.")
                return

        # Step 2: Extract frames
        status.info("Step 2/8 — Extracting frames…")
        progress.progress(0.15, text="Extracting frames…")
        frames = extract_frames(norm_path)
        original_count = len(frames)
        if original_count == 0:
            st.error("❌ No frames decoded from video.")
            return

        # Step 3: Trim
        status.info("Step 3/8 — Trimming idle segments…")
        progress.progress(0.20, text="Trimming…")
        frames, trim_start, trim_end = trim_idle_segments(frames, cfg)
        trimmed_count = len(frames)

        # Step 4: Crop
        status.info("Step 4/8 — Localizing signer & cropping…")
        progress.progress(0.30, text="Signer localization…")
        bbox = compute_signer_bbox(frames, cfg)
        frames = crop_and_resize(frames, bbox, cfg.crop_short_side)
        crop_h, crop_w = frames[0].shape[:2]

        # Step 5: Denoise
        status.info("Step 5/8 — Denoising frames…")
        progress.progress(0.40, text="Denoising…")
        frames = denoise_frames(frames, cfg)

        # Step 6: Keypoints
        status.info("Step 6/8 — Extracting keypoints (MediaPipe)…")
        progress.progress(0.50, text="Keypoint extraction…")
        keypoints, kp_confidence = extract_keypoints(frames, cfg)

        # Step 7: Optical flow
        flow_vectors = None
        flow_magnitudes = None
        if not skip_flow:
            status.info(f"Step 7/8 — Computing optical flow (RAFT-{cfg.flow_model})…")
            progress.progress(0.65, text="Optical flow (RAFT)…")
            flow_ext = FlowExtractor(cfg)
            flow_vectors, flow_magnitudes = flow_ext.extract(frames)
        else:
            progress.progress(0.65, text="Optical flow skipped")

        # Step 8: Masks + normalize + save
        status.info("Step 8/8 — Generating masks, normalizing, saving…")
        progress.progress(0.80, text="Masks & saving…")
        masks = generate_masks(keypoints, flow_magnitudes, crop_h, crop_w, cfg)

        data = normalize_sequence_length(
            frames, keypoints, flow_vectors, flow_magnitudes, masks, cfg
        )

        metadata = {
            "source_video": video_name,
            "label": label,
            "original_frame_count": original_count,
            "trimmed_frame_count": trimmed_count,
            "trim_range": [trim_start, trim_end],
            "crop_bbox": list(bbox),
            "crop_size": [crop_h, crop_w],
            "keypoint_confidence_mean": round(kp_confidence, 4),
            "target_fps": cfg.target_fps,
            "target_sequence_length": cfg.target_sequence_length,
            "flow_computed": not skip_flow,
            "pipeline_version": "1.0.0",
        }
        sample_dir = save_sample(cfg.output_dir, label, video_name, data, metadata)

        # Quality check
        report = check_sample(sample_dir)
        progress.progress(1.0, text="Done!")

        if report.passed:
            status.success(
                f"✅ Preprocessing complete — saved to `{sample_dir}`  "
                f"({trimmed_count} frames trimmed from {original_count}, "
                f"keypoint confidence: {kp_confidence:.2f})"
            )
        else:
            status.warning(
                f"⚠️ Saved to `{sample_dir}` with warnings: {', '.join(report.warnings)}"
            )

        # Store preview
        st.session_state[_key("preview_frames")] = data["frames"]
        st.session_state[_key("preview_keypoints")] = data["keypoints"]
        st.session_state[_key("preview_flow_mag")] = data.get("flow_magnitudes")
        st.session_state[_key("preview_masks")] = data.get("masks")
        st.session_state[_key("preview_metadata")] = metadata

    except Exception as exc:
        progress.progress(1.0, text="Error")
        status.error(f"❌ Pipeline failed: {exc}")
        st.exception(exc)


def _load_preview_from_disk(cfg: PipelineConfig, label: str, video_name: str):
    """Load previously saved outputs for preview."""
    stem = Path(video_name).stem
    sample_dir = cfg.output_dir / label / stem

    frames_p = sample_dir / "frames.npy"
    kp_p = sample_dir / "keypoints.npy"
    meta_p = sample_dir / "metadata.json"

    if frames_p.exists():
        st.session_state[_key("preview_frames")] = np.load(frames_p)
    if kp_p.exists():
        st.session_state[_key("preview_keypoints")] = np.load(kp_p)

    flow_mag_p = sample_dir / "flow_magnitudes.npy"
    if flow_mag_p.exists():
        st.session_state[_key("preview_flow_mag")] = np.load(flow_mag_p)
    masks_p = sample_dir / "masks.npy"
    if masks_p.exists():
        st.session_state[_key("preview_masks")] = np.load(masks_p)

    if meta_p.exists():
        with open(meta_p, encoding="utf-8") as f:
            st.session_state[_key("preview_metadata")] = json.load(f)


def _render_preview():
    """Display preview of preprocessed output."""
    st.divider()
    st.subheader("🔍 Preview")

    meta = st.session_state[_key("preview_metadata")]
    if meta:
        mcols = st.columns(5)
        mcols[0].metric("Original Frames", meta.get("original_frame_count", "—"))
        mcols[1].metric("After Trim", meta.get("trimmed_frame_count", "—"))
        crop = meta.get("crop_size", ["—", "—"])
        mcols[2].metric("Crop Size", f"{crop[0]}×{crop[1]}")
        mcols[3].metric(
            "KP Confidence", f"{meta.get('keypoint_confidence_mean', 0):.2f}"
        )
        mcols[4].metric("Flow Computed", "Yes" if meta.get("flow_computed") else "No")

    frames = st.session_state[_key("preview_frames")]
    if frames is not None and len(frames) > 0:
        tab_frm, tab_flow, tab_mask = st.tabs(["Frames", "Optical Flow", "Masks"])

        with tab_frm:
            st.markdown("**Cropped & denoised frames** (uniformly sampled):")
            sample_idx = list(range(0, len(frames), max(1, len(frames) // 8)))[:8]
            cols = st.columns(len(sample_idx))
            for i, idx in enumerate(sample_idx):
                with cols[i]:
                    st.image(frames[idx], caption=f"#{idx}", use_container_width=True)

        with tab_flow:
            flow_mag = st.session_state[_key("preview_flow_mag")]
            if flow_mag is not None and len(flow_mag) > 0:
                st.markdown("**Optical flow magnitude** (brighter = more motion):")
                sample_idx = list(range(0, len(flow_mag), max(1, len(flow_mag) // 8)))[
                    :8
                ]
                cols = st.columns(len(sample_idx))
                for i, idx in enumerate(sample_idx):
                    with cols[i]:
                        mag = flow_mag[idx]
                        if mag.max() > 0:
                            viz = (mag / mag.max() * 255).astype(np.uint8)
                        else:
                            viz = np.zeros_like(mag, dtype=np.uint8)
                        st.image(viz, caption=f"#{idx}", use_container_width=True)
            else:
                st.info("Optical flow was skipped for this sample.")

        with tab_mask:
            masks = st.session_state[_key("preview_masks")]
            if masks is not None and len(masks) > 0:
                st.markdown("**Motion masks** (white = signer region):")
                sample_idx = list(range(0, len(masks), max(1, len(masks) // 8)))[:8]
                cols = st.columns(len(sample_idx))
                for i, idx in enumerate(sample_idx):
                    with cols[i]:
                        st.image(
                            masks[idx], caption=f"#{idx}", use_container_width=True
                        )
            else:
                st.info("No masks available.")

    if meta:
        with st.expander("📋 Full Metadata"):
            st.json(meta)


# ── Batch mode ───────────────────────────────────────────────────────────────


def _render_batch(cfg: PipelineConfig):
    st.subheader("📦 Batch Process — Full Dataset")

    labels_path = cfg.labels_csv
    if not labels_path.exists():
        st.error(f"labels.csv not found at `{labels_path}`")
        return

    rows = _read_labels(labels_path)
    total = len(rows)

    # Dataset overview
    label_counts: dict[str, int] = {}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    st.markdown(f"**{total} videos** across **{len(label_counts)} classes**")
    with st.expander("Class distribution"):
        for cls, cnt in sorted(label_counts.items()):
            st.text(f"  {cls:15s}  {cnt} videos")

    col_a, col_b = st.columns(2)
    with col_a:
        skip_flow = st.checkbox(
            "Skip optical flow (much faster)",
            value=False,
            key=_key("batch_skip_flow"),
        )
    with col_b:
        skip_existing = st.checkbox(
            "Skip already processed",
            value=True,
            key=_key("batch_skip_existing"),
        )

    if st.button("🚀 Start Batch Processing", type="primary", key=_key("run_batch")):
        _run_batch(cfg, rows, skip_flow, skip_existing)

    results = st.session_state.get(_key("batch_results"))
    if results:
        _render_batch_results(results)


def _run_batch(
    cfg: PipelineConfig,
    rows: list[dict],
    skip_flow: bool,
    skip_existing: bool,
):
    """Process every video in labels.csv with live progress."""
    from app.preprocessing.runner import PreprocessingPipeline

    pipeline = PreprocessingPipeline(cfg)
    total = len(rows)

    progress = st.progress(0.0, text=f"Processing 0/{total}…")
    status_counts: dict[str, int] = {"ok": 0, "skipped": 0, "error": 0}
    all_results: list[dict] = []

    for idx, row in enumerate(rows):
        video_name = row["video_name"]
        label = row["label"]
        rel_path = row["path"]
        video_path = cfg.dataset_dir / rel_path

        if not video_path.exists():
            result = {
                "status": "error",
                "video": video_name,
                "error": "file not found",
            }
        else:
            result = pipeline.process_single_video(
                video_path,
                label,
                video_name,
                skip_existing=skip_existing,
                skip_flow=skip_flow,
            )

        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        all_results.append(result)

        frac = (idx + 1) / total
        icon = {"ok": "✓", "skipped": "⊘", "error": "✗"}.get(result["status"], "?")
        progress.progress(frac, text=f"{icon} [{idx + 1}/{total}] {label}/{video_name}")

    progress.progress(1.0, text="Batch complete!")

    st.session_state[_key("batch_results")] = {
        "total": total,
        "processed": status_counts.get("ok", 0),
        "skipped": status_counts.get("skipped", 0),
        "errors": status_counts.get("error", 0),
        "results": all_results,
    }


def _render_batch_results(summary: dict):
    st.divider()
    st.subheader("📊 Batch Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", summary["total"])
    c2.metric("Processed", summary["processed"])
    c3.metric("Skipped", summary["skipped"])
    c4.metric("Errors", summary["errors"])

    errors = [r for r in summary["results"] if r["status"] == "error"]
    if errors:
        with st.expander(f"⚠️ {len(errors)} errors", expanded=True):
            for e in errors:
                st.text(f"  ✗ {e['video']}: {e.get('error', 'unknown')}")

    warnings = [
        r
        for r in summary["results"]
        if r["status"] == "ok" and r.get("report", {}).get("warnings")
    ]
    if warnings:
        with st.expander(f"⚠️ {len(warnings)} samples with warnings"):
            for w in warnings:
                st.text(f"  ⚠ {w['video']}: {', '.join(w['report']['warnings'])}")


# ── Quality report mode ──────────────────────────────────────────────────────


def _render_report(cfg: PipelineConfig):
    st.subheader("📊 Quality Report")

    if not cfg.output_dir.exists():
        st.info("No preprocessed data found. Run the pipeline first.")
        return

    if st.button("🔄 Generate Report", key=_key("gen_report")):
        with st.spinner("Scanning preprocessed data…"):
            report = generate_dataset_report(cfg.output_dir)
        st.session_state[_key("report_data")] = report

    report = st.session_state.get(_key("report_data"))
    if not report:
        st.info("Click 'Generate Report' to scan preprocessed data.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", report["total_samples"])
    c2.metric("Passed", report["passed"])
    c3.metric("Failed", report["failed_count"])

    st.markdown("**Per-class counts:**")
    if report["class_counts"]:
        ncols = min(5, len(report["class_counts"]))
        cols = st.columns(ncols)
        for i, (cls, cnt) in enumerate(sorted(report["class_counts"].items())):
            cols[i % ncols].metric(cls, cnt)

    if report["failed_samples"]:
        st.markdown("---")
        st.markdown(f"**⚠️ {len(report['failed_samples'])} failed samples:**")
        for f in report["failed_samples"]:
            st.text(f"  ✗ {f['label']}/{f['video']}: {'; '.join(f['warnings'])}")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_labels(labels_path: Path) -> list[dict]:
    with open(labels_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
