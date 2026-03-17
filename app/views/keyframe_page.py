"""
Keyframe Extractor Page

Streamlit UI for the Farneback + MediaPipe hybrid keyframe
extraction pipeline (ported from asl_keyframe_extractor CLI tool).

Supports:
    - Single video upload → extract → preview → save
    - Batch processing of a dataset folder → organised output
"""

import os
import tempfile

import cv2
import numpy as np
import streamlit as st

from app.core.keyframe_pipeline import (
    PipelineConfig,
    PipelineResult,
    discover_videos,
    run_batch,
    run_pipeline,
    save_individual_keyframes,
)
from ._folder_browser import folder_input_with_browse


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────


def _init_state():
    defaults = {
        "kf_result": None,
        "kf_video_name": "",
        "kf_output_folder": "",
        "kf_batch_input": "",
        "kf_batch_output": "",
        "kf_batch_results": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────
# Parameter Controls
# ─────────────────────────────────────────────


def _render_parameters() -> PipelineConfig:
    """Render pipeline parameter controls and return a PipelineConfig."""

    with st.expander("⚙️ Pipeline Parameters", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            min_kf = st.number_input(
                "Min keyframes",
                min_value=2,
                max_value=50,
                value=8,
                step=1,
                key="kf_min",
            )
            max_kf = st.number_input(
                "Max keyframes",
                min_value=2,
                max_value=50,
                value=15,
                step=1,
                key="kf_max",
            )

        with col2:
            smooth_sigma = st.number_input(
                "Smooth sigma",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                key="kf_sigma",
                help="Gaussian smoothing on motion signal. Lower = sharper peaks.",
            )
            hold_thresh = st.number_input(
                "Hold threshold",
                min_value=0.01,
                max_value=0.5,
                value=0.12,
                step=0.01,
                key="kf_hold",
                help="Sandwiched hold sensitivity. Lower = catches subtler holds.",
            )

        with col3:
            mp_complexity = st.selectbox(
                "MediaPipe complexity",
                options=[0, 1, 2],
                index=1,
                key="kf_mp",
                format_func=lambda x: {
                    0: "0 — Fast",
                    1: "1 — Balanced",
                    2: "2 — Accurate",
                }[x],
            )

    return PipelineConfig(
        min_keyframes=min_kf,
        max_keyframes=max(max_kf, min_kf),
        smooth_sigma=smooth_sigma,
        sandwiched_hold_threshold=hold_thresh,
        mediapipe_complexity=mp_complexity,
    )


# ─────────────────────────────────────────────
# Single Video Mode
# ─────────────────────────────────────────────


def _render_single_mode(config: PipelineConfig):
    """Upload a video → extract keyframes → preview → save."""

    uploaded = st.file_uploader(
        "Upload a sign language video",
        type=["mp4", "mov", "avi", "mkv"],
        key="kf_upload",
    )

    if uploaded is None:
        return

    # Save upload to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.close()
    video_path = tfile.name

    try:
        # Extract button
        if st.button("🚀 Extract Keyframes", type="primary", key="kf_extract"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _on_progress(frac, msg):
                progress_bar.progress(min(frac, 1.0))
                status_text.text(msg)

            with st.spinner("Running extraction pipeline…"):
                result = run_pipeline(video_path, config, progress_cb=_on_progress)
                result.video_stem = uploaded.name.rsplit(".", 1)[0]

            progress_bar.empty()
            status_text.empty()

            st.session_state["kf_result"] = result
            st.session_state["kf_video_name"] = uploaded.name
            st.success(
                f"✅ Extracted **{len(result.keyframe_indices)}** keyframes "
                f"from {result.total_frames} frames "
                f"({result.fps:.1f} fps, "
                f"hands in {result.hands_detected}/{result.total_frames} frames)"
            )

        # Display results
        result: PipelineResult = st.session_state["kf_result"]
        if result is not None:
            _render_results(result)
            _render_save_section(result)

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


def _render_results(result: PipelineResult):
    """Display extraction results: metrics, table, and frame preview."""

    st.divider()
    st.subheader("📊 Extraction Results")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Keyframes", len(result.keyframe_indices))
    col2.metric("Total Frames", result.total_frames)
    col3.metric("FPS", f"{result.fps:.1f}")
    col4.metric("Hands Detected", f"{result.hands_detected}/{result.total_frames}")

    # Results table
    with st.expander("📋 Keyframe Details", expanded=True):
        rows = []
        for rank, idx in enumerate(result.keyframe_indices):
            rows.append(
                {
                    "#": rank + 1,
                    "Frame": idx,
                    "Time (s)": result.keyframe_times[rank],
                    "Transition": round(float(result.fused_transition[idx]), 3),
                    "Hold": round(float(result.fused_hold[idx]), 3),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # Frame preview grid
    st.subheader("🖼️ Keyframe Preview")
    cols = st.columns(5)
    for i, bgr in enumerate(result.keyframe_images):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with cols[i % 5]:
            st.image(
                rgb,
                caption=f"#{i + 1}  Frame {result.keyframe_indices[i]}  ({result.keyframe_times[i]}s)",
                use_container_width=True,
            )


def _render_save_section(result: PipelineResult):
    """Save extracted keyframes to a folder."""

    st.divider()
    st.subheader("💾 Save Keyframes")

    folder_input_with_browse(
        "Output folder path",
        session_key="kf_output_folder",
        placeholder="./outputs/keyframes",
        dialog_title="Select Output Folder",
    )

    if st.button("💾 Save Individual Frames", key="kf_save"):
        out_dir = st.session_state.get("kf_output_folder") or "./outputs/keyframes"

        # Create subfolder named after the video
        video_dir = os.path.join(out_dir, result.video_stem)
        paths = save_individual_keyframes(
            result.keyframe_images,
            list(range(len(result.keyframe_images))),
            video_dir,
        )
        st.success(f"✅ Saved {len(paths)} keyframe images to `{video_dir}`")


# ─────────────────────────────────────────────
# Batch Processing Mode
# ─────────────────────────────────────────────


def _render_batch_mode(config: PipelineConfig):
    """Process all videos in a dataset folder."""

    st.markdown("**Expected folder structure:**  `dataset_folder / label / video.mp4`")

    folder_input_with_browse(
        "Dataset folder path",
        session_key="kf_batch_input",
        placeholder="./dataset/raw_video_data",
        dialog_title="Select Dataset Folder",
    )

    folder_input_with_browse(
        "Output folder path",
        session_key="kf_batch_output",
        placeholder="./outputs/keyframes",
        dialog_title="Select Output Folder",
    )

    dataset_dir = st.session_state.get("kf_batch_input", "")
    output_dir = st.session_state.get("kf_batch_output", "") or "./outputs/keyframes"

    if not dataset_dir:
        st.info(
            "Paste or browse to the dataset folder containing label subdirectories."
        )
        return

    # Show video count preview
    try:
        videos = discover_videos(dataset_dir)
        labels = set(v[1] for v in videos)
        st.caption(
            f"📁 Found **{len(videos)}** videos across **{len(labels)}** labels: "
            f"{', '.join(sorted(labels)[:10])}{'…' if len(labels) > 10 else ''}"
        )
    except Exception as e:
        st.error(f"Could not scan dataset: {e}")
        return

    if not videos:
        st.warning("No video files found in the dataset folder.")
        return

    # Process button
    if st.button("🚀 Process All Videos", type="primary", key="kf_batch_run"):
        progress = st.progress(0)
        status = st.empty()
        log_area = st.empty()

        log_lines = []

        def _on_video(idx, total, label, stem):
            progress.progress((idx) / total)
            status.text(f"[{idx + 1}/{total}]  {label}/{stem}")

        results = []

        # Run batch with per-video callback
        for i, (video_path, label, video_stem) in enumerate(videos):
            _on_video(i, len(videos), label, video_stem)

            out_dir = os.path.join(output_dir, label, video_stem)

            try:
                result = run_pipeline(video_path, config)
                save_individual_keyframes(
                    result.keyframe_images,
                    list(range(len(result.keyframe_images))),
                    out_dir,
                )
                entry = {
                    "Label": label,
                    "Video": video_stem,
                    "Status": "✓",
                    "Keyframes": len(result.keyframe_indices),
                    "Error": "",
                }
            except Exception as e:
                entry = {
                    "Label": label,
                    "Video": video_stem,
                    "Status": "✗",
                    "Keyframes": 0,
                    "Error": str(e),
                }

            results.append(entry)
            log_lines.append(
                f"{entry['Status']}  {label}/{video_stem}  →  {entry['Keyframes']} keyframes"
            )
            log_area.text("\n".join(log_lines[-15:]))  # show last 15 lines

        progress.progress(1.0)
        status.empty()
        log_area.empty()

        st.session_state["kf_batch_results"] = results

        success = sum(1 for r in results if r["Status"] == "✓")
        st.success(
            f"✅ Batch complete: **{success}/{len(results)}** videos processed successfully"
        )

    # Display batch results
    if st.session_state.get("kf_batch_results"):
        st.divider()
        st.subheader("📊 Batch Results")
        st.dataframe(
            st.session_state["kf_batch_results"],
            use_container_width=True,
            hide_index=True,
        )


# ─────────────────────────────────────────────
# Main Render
# ─────────────────────────────────────────────


def render():
    """Render the keyframe extractor page."""
    _init_state()

    st.caption(
        "Extract keyframes from sign language videos using the "
        "Farneback + MediaPipe hybrid pipeline."
    )

    config = _render_parameters()

    mode = st.radio(
        "Mode",
        ["📹 Single Video", "📂 Batch Processing"],
        horizontal=True,
        key="kf_mode",
    )

    st.divider()

    if mode == "📹 Single Video":
        _render_single_mode(config)
    else:
        _render_batch_mode(config)
