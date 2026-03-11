"""
Keyframe Extractor Page

Extract keyframes from sign language videos for dataset preparation.
"""

import os
import tempfile

import streamlit as st

from app.core.video_utils import (
    get_frames_from_video,
    get_video_info,
    preprocess_frames,
    VALID_SIZES,
)
from app.core.algorithms import ALGORITHM_MAP, ALGORITHM_NAMES
from app.core.file_utils import save_frames_to_folder
from ._folder_browser import folder_input_with_browse


def init_session_state():
    """Initialize session state variables for keyframe extractor."""
    if "kf_extracted_frames" not in st.session_state:
        st.session_state["kf_extracted_frames"] = None
    if "kf_extracted_indices" not in st.session_state:
        st.session_state["kf_extracted_indices"] = None
    if "kf_video_name" not in st.session_state:
        st.session_state["kf_video_name"] = ""
    if "kf_output_folder" not in st.session_state:
        st.session_state["kf_output_folder"] = ""


def render():
    """Render the keyframe extractor page."""
    init_session_state()

    st.caption("Extract keyframes from sign language videos for dataset preparation.")

    # --- ALGORITHM SELECTOR ---
    keyframe_algo_choice = st.selectbox(
        "Keyframe Extraction Algorithm", ALGORITHM_NAMES
    )

    # --- TARGET FRAME COUNT ---
    num_frames_target = st.number_input(
        "Target Number of Frames",
        min_value=2,
        max_value=500,
        value=10,
        step=1,
        key="kf_num_frames",
        help="Enter any value from 2 upward — type it in directly.",
    )

    # --- FILE UPLOAD ---
    uploaded_file = st.file_uploader(
        "Upload a Sign Language Video", type=["mp4", "mov", "avi"]
    )

    if uploaded_file is None:
        return

    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name

    # ── VIDEO INFO ──────────────────────────────────────────────────────────
    info = get_video_info(video_path)
    st.caption(
        f"📹 Native: {info['width']}×{info['height']}  |  "
        f"{info['fps']:.1f} fps  |  {info['frame_count']} frames"
    )

    # ── PREPROCESSING OPTIONS ───────────────────────────────────────────────
    with st.expander("⚙️ Video Preprocessing", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Spatial Rescaling**")
            rescale_choice = st.selectbox(
                "Resize all frames to",
                options=["No rescaling"] + list(VALID_SIZES.keys()),
                key="kf_rescale",
                help="Downsample frames before extraction. Reduces memory & speeds up RAFT.",
            )

            st.markdown("**FPS Resampling**")
            resample_fps = st.checkbox(
                "Resample to 30 fps",
                value=True,
                key="kf_resample_fps",
                help="Skip frames so the video is treated as exactly 30 fps. "
                "Videos already at ≤30 fps are unaffected.",
            )

        with col2:
            st.markdown("**Contrast Enhancement**")
            run_contrast = st.checkbox(
                "Pixel contrast normalisation",
                value=False,
                key="kf_contrast",
                help="Min-max stretch each frame to full [0, 255] range.",
            )
            run_clahe = st.checkbox(
                "CLAHE (adaptive histogram equalisation)",
                value=False,
                key="kf_clahe",
                help="Enhances local contrast while preserving colour (LAB L-channel).",
            )
            if run_clahe:
                clahe_clip = st.slider(
                    "CLAHE clip limit",
                    min_value=1.0,
                    max_value=8.0,
                    value=2.0,
                    step=0.5,
                    key="kf_clahe_clip",
                )
            else:
                clahe_clip = 2.0

    target_fps = 30.0 if resample_fps else 0

    try:
        frames = get_frames_from_video(video_path, target_fps=target_fps)

        if not frames:
            st.error("Could not extract frames.")
            return

        # Apply preprocessing pipeline
        rescale_size = (
            VALID_SIZES.get(rescale_choice)
            if rescale_choice != "No rescaling"
            else None
        )
        frames = preprocess_frames(
            frames,
            rescale_size=rescale_size,
            run_contrast_norm=run_contrast,
            run_clahe=run_clahe,
            clahe_clip=clahe_clip,
        )

        pre_info = []
        if rescale_size:
            pre_info.append(f"resized to {rescale_size[0]}×{rescale_size[1]}")
        if resample_fps:
            pre_info.append("30 fps")
        if run_contrast:
            pre_info.append("contrast normalised")
        if run_clahe:
            pre_info.append(f"CLAHE clip={clahe_clip}")
        if pre_info:
            st.success(
                f"✅ Preprocessing applied: {', '.join(pre_info)}  → {len(frames)} frames"
            )

        st.subheader(f"Preprocessed Video ({len(frames)} frames)")
        st.image(
            frames[len(frames) // 2],
            caption="Middle Frame Preview (after preprocessing)",
            width=400,
        )

        # --- EXTRACTION BUTTON ---
        if st.button("Extract Keyframes"):
            _extract_keyframes(
                frames, keyframe_algo_choice, num_frames_target, uploaded_file.name
            )

        # --- DISPLAY & ACTIONS (same for every algorithm) ---
        if st.session_state["kf_extracted_frames"] is not None:
            _display_extracted_frames(keyframe_algo_choice)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


# =============================================================================
# Keyframe extraction helpers
# =============================================================================


def _extract_keyframes(frames, algo_choice, num_frames_target, video_name):
    """Extract keyframes using the selected algorithm."""
    st.session_state["kf_video_name"] = video_name

    if algo_choice in ALGORITHM_MAP:
        algo_fn = ALGORITHM_MAP[algo_choice]
        selected, idxs = algo_fn(frames, num_frames_target)
        st.session_state["kf_extracted_frames"] = selected
        st.session_state["kf_extracted_indices"] = idxs
        st.success(f"Extracted {len(selected)} frames using {algo_choice}.")


def _display_extracted_frames(algo_choice):
    """Display extracted frames and save options."""
    st.divider()

    # --- SAVE FRAMES BUTTON ---
    folder_input_with_browse(
        "Output folder path for keyframes",
        session_key="kf_output_folder",
        placeholder="/absolute/path/to/output",
        dialog_title="Select Output Folder for Keyframes",
    )

    if st.button("💾 Save Frames to Folder"):
        if algo_choice == "Relative Quantization (Paper Implementation)":
            st.warning(
                "The Paper Implementation mode is for visualization only. "
                "Switch to Uniform or Motion to save raw data for training."
            )
        else:
            success, msg = save_frames_to_folder(
                st.session_state["kf_extracted_frames"],
                st.session_state["kf_extracted_indices"],
                st.session_state["kf_video_name"],
                target_dir=st.session_state.get("kf_output_folder") or None,
            )
            if success:
                st.success(msg)
            else:
                st.error(msg)

    # --- PREVIEW SECTION ---
    st.divider()
    st.write("**Extracted Keyframes Preview:**")
    cols = st.columns(5)
    for i, frame in enumerate(st.session_state["kf_extracted_frames"]):
        with cols[i % 5]:
            st.image(
                frame,
                caption=f"Frame {st.session_state['kf_extracted_indices'][i]}",
                use_container_width=True,
            )
