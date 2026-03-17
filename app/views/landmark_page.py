"""
Landmark Extractor Page

Streamlit UI for the MediaPipe Holistic → RQ landmark extraction pipeline
(ported from asl_landmark_extractor CLI tool).

Supports:
    - Single directory of keyframes → extract → preview → save .npy
    - Batch processing of an organised keyframe dataset → .npy per video
"""

import os

import numpy as np
import streamlit as st

from app.core.landmark_pipeline import (
    LandmarkConfig,
    LandmarkResult,
    discover_frame_dirs,
    run_batch,
    run_pipeline,
    save_landmarks,
)
from ._folder_browser import folder_input_with_browse


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "lm_result": None,
        "lm_input_folder": "",
        "lm_output_folder": "",
        "lm_batch_input": "",
        "lm_batch_output": "",
        "lm_batch_results": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─────────────────────────────────────────────
# Parameter Controls
# ─────────────────────────────────────────────

def _render_parameters() -> LandmarkConfig:
    """Render pipeline parameter controls and return a LandmarkConfig."""

    with st.expander("⚙️ Pipeline Parameters", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            target_len = st.number_input(
                "Target sequence length", min_value=5, max_value=100,
                value=15, step=1, key="lm_target_len",
                help="Pad or truncate to this fixed length.",
            )
            mp_complexity = st.selectbox(
                "MediaPipe complexity",
                options=[0, 1, 2],
                index=1,
                key="lm_mp",
                format_func=lambda x: {
                    0: "0 — Fast", 1: "1 — Balanced", 2: "2 — Accurate"
                }[x],
            )

        with col2:
            scale_factor = st.number_input(
                "Scale factor", min_value=1.0, max_value=1000.0,
                value=100.0, step=10.0, key="lm_scale",
                help="Post-RQ feature scaling (default ×100).",
            )

        with col3:
            st.markdown("**RQ quantization levels**")
            st.caption("Hand: (10, 10, 5) · Face: (5, 5, 3) · Pose: (10, 10, 5)")
            st.caption("These match the research paper defaults.")

    return LandmarkConfig(
        target_sequence_length=target_len,
        mediapipe_complexity=mp_complexity,
        scale_factor=scale_factor,
    )


# ─────────────────────────────────────────────
# Single Directory Mode
# ─────────────────────────────────────────────

def _render_single_mode(config: LandmarkConfig):
    """Select frames folder → extract landmarks → preview → save .npy."""

    st.markdown("Select a folder containing extracted keyframe images (`frame_0.png`, `frame_1.png`, …)")

    folder_input_with_browse(
        "Keyframes folder path",
        session_key="lm_input_folder",
        placeholder="./output/keyframes/label/video_stem",
        dialog_title="Select Keyframes Folder",
    )

    frames_dir = st.session_state.get("lm_input_folder", "")

    if not frames_dir:
        st.info("Paste or browse to a folder of keyframe images.")
        return

    if not os.path.isdir(frames_dir):
        st.error(f"Folder not found: `{frames_dir}`")
        return

    # Extract button
    if st.button("🚀 Extract Landmarks", type="primary", key="lm_extract"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def _on_progress(frac, msg):
            progress_bar.progress(min(frac, 1.0))
            status_text.text(msg)

        with st.spinner("Running RQ landmark pipeline…"):
            result = run_pipeline(frames_dir, config, progress_cb=_on_progress)

        progress_bar.empty()
        status_text.empty()

        st.session_state["lm_result"] = result
        st.success(
            f"✅ Extracted landmarks: shape **{result.landmarks.shape}** "
            f"from {result.original_frame_count} keyframes "
            f"(dominant: {result.dominant_hand} hand)"
        )

    # Display results
    result: LandmarkResult = st.session_state.get("lm_result")
    if result is not None:
        _render_results(result)
        _render_save_section(result)


def _render_results(result: LandmarkResult):
    """Display extraction results: metrics, data preview."""

    st.divider()
    st.subheader("📊 Extraction Results")

    # Metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Shape", f"{result.landmarks.shape}")
    c2.metric("Input Frames", result.original_frame_count)
    c3.metric("Dominant Hand", result.dominant_hand.title())

    non_zero = int(np.sum(np.any(result.landmarks != 0, axis=1)))
    c4.metric("Non-zero Frames", f"{non_zero}/{result.landmarks.shape[0]}")

    has_issues = np.isnan(result.landmarks).any() or np.isinf(result.landmarks).any()
    c5.metric("NaN / Inf", "⚠️ YES" if has_issues else "✓ None")

    # Detection breakdown
    with st.expander("🔍 Detection Breakdown"):
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Left Hand", f"{result.hands_detected_left}/{result.original_frame_count}")
        dc2.metric("Right Hand", f"{result.hands_detected_right}/{result.original_frame_count}")
        dc3.metric("Face", f"{result.face_detected}/{result.original_frame_count}")

    # Value range
    vals = result.landmarks[result.landmarks != 0]
    if len(vals) > 0:
        st.caption(f"Value range: [{vals.min():.1f}, {vals.max():.1f}]")

    # Data preview
    with st.expander("📋 Preview Landmark Data"):
        st.dataframe(
            result.landmarks,
            use_container_width=True,
            height=300,
        )


def _render_save_section(result: LandmarkResult):
    """Save landmarks as .npy file."""

    st.divider()
    st.subheader("💾 Save Landmarks")

    folder_input_with_browse(
        "Output folder path",
        session_key="lm_output_folder",
        placeholder="./output/landmarks",
        dialog_title="Select Output Folder",
    )

    if st.button("💾 Save as .npy", key="lm_save"):
        out_dir = st.session_state.get("lm_output_folder") or "./output/landmarks"
        out_path = os.path.join(out_dir, f"{result.video_stem}.npy")
        saved = save_landmarks(result.landmarks, out_path)
        st.success(f"✅ Saved → `{saved}`  (shape: {result.landmarks.shape})")


# ─────────────────────────────────────────────
# Batch Processing Mode
# ─────────────────────────────────────────────

def _render_batch_mode(config: LandmarkConfig):
    """Process all keyframe directories in a dataset."""

    st.markdown(
        "**Expected folder structure:**  "
        "`root / label / video_stem / frame_0.png, frame_1.png, …`"
    )

    folder_input_with_browse(
        "Keyframes root folder",
        session_key="lm_batch_input",
        placeholder="./output/keyframes",
        dialog_title="Select Keyframes Root Folder",
    )

    folder_input_with_browse(
        "Output folder for .npy files",
        session_key="lm_batch_output",
        placeholder="./output/landmarks",
        dialog_title="Select Output Folder",
    )

    root_dir   = st.session_state.get("lm_batch_input", "")
    output_dir = st.session_state.get("lm_batch_output", "") or "./output/landmarks"

    if not root_dir:
        st.info("Paste or browse to the root folder of extracted keyframes.")
        return

    # Preview what will be processed
    try:
        frame_dirs = discover_frame_dirs(root_dir)
        labels = set(d[1] for d in frame_dirs)
        st.caption(
            f"📁 Found **{len(frame_dirs)}** video keyframe directories "
            f"across **{len(labels)}** labels: "
            f"{', '.join(sorted(labels)[:10])}{'…' if len(labels) > 10 else ''}"
        )
    except Exception as e:
        st.error(f"Could not scan directory: {e}")
        return

    if not frame_dirs:
        st.warning("No keyframe directories found.")
        return

    # Process button
    if st.button("🚀 Process All Landmarks", type="primary", key="lm_batch_run"):
        progress = st.progress(0)
        status = st.empty()
        log_area = st.empty()

        log_lines = []

        results = []

        for i, (frames_dir, label, video_stem) in enumerate(frame_dirs):
            progress.progress(i / len(frame_dirs))
            status.text(f"[{i + 1}/{len(frame_dirs)}]  {label}/{video_stem}")

            output_path = os.path.join(output_dir, label, f"{video_stem}.npy")

            try:
                result = run_pipeline(frames_dir, config)
                save_landmarks(result.landmarks, output_path)
                entry = {
                    "Label": label,
                    "Video": video_stem,
                    "Status": "✓",
                    "Shape": f"({result.landmarks.shape[0]}, {result.landmarks.shape[1]})",
                    "Dominant": result.dominant_hand,
                    "Error": "",
                }
            except Exception as e:
                entry = {
                    "Label": label,
                    "Video": video_stem,
                    "Status": "✗",
                    "Shape": "",
                    "Dominant": "",
                    "Error": str(e),
                }

            results.append(entry)
            log_lines.append(
                f"{entry['Status']}  {label}/{video_stem}  →  {entry['Shape']}"
            )
            log_area.text("\n".join(log_lines[-15:]))

        progress.progress(1.0)
        status.empty()
        log_area.empty()

        st.session_state["lm_batch_results"] = results

        success = sum(1 for r in results if r["Status"] == "✓")
        st.success(
            f"✅ Batch complete: **{success}/{len(results)}** processed successfully"
        )

    # Display batch results
    if st.session_state.get("lm_batch_results"):
        st.divider()
        st.subheader("📊 Batch Results")
        st.dataframe(
            st.session_state["lm_batch_results"],
            use_container_width=True,
            hide_index=True,
        )


# ─────────────────────────────────────────────
# Main Render
# ─────────────────────────────────────────────

def render():
    """Render the landmark extractor page."""
    _init_state()

    st.caption(
        "Extract and process MediaPipe landmarks from keyframe images "
        "using the Relative Quantization (RQ) pipeline."
    )

    config = _render_parameters()

    mode = st.radio(
        "Mode",
        ["📂 Single Directory", "📦 Batch Processing"],
        horizontal=True,
        key="lm_mode",
    )

    st.divider()

    if mode == "📂 Single Directory":
        _render_single_mode(config)
    else:
        _render_batch_mode(config)
