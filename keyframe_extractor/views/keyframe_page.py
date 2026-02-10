"""
Keyframe Extractor Page

Extract keyframes from sign language videos for dataset preparation.
"""

import os
import tempfile

import streamlit as st

from video_utils import get_frames_from_video
from algorithms import (
    ALGORITHM_MAP,
    ALGORITHM_NAMES,
    uniform_sampling,
    draw_quantization_grid,
)
from file_utils import save_frames_to_folder


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

    # --- CONTROLS ---
    keyframe_algo_choice = st.selectbox(
        "Keyframe Extraction Algorithm", ALGORITHM_NAMES
    )
    num_frames_target = st.slider(
        "Target Number of Frames", 10, 50, 30, key="kf_num_frames"
    )

    # --- FILE UPLOAD ---
    uploaded_file = st.file_uploader(
        "Upload a Sign Language Video", type=["mp4", "mov", "avi"]
    )

    if uploaded_file is None:
        return

    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name

    try:
        frames = get_frames_from_video(video_path)

        if not frames:
            st.error("Could not extract frames.")
            return

        st.subheader(f"Original Video ({len(frames)} frames)")
        st.image(frames[len(frames) // 2], caption="Middle Frame Preview", width=400)

        # --- EXTRACTION BUTTON ---
        if st.button("Extract Keyframes"):
            _extract_keyframes(
                frames, keyframe_algo_choice, num_frames_target, uploaded_file.name
            )

        # --- DISPLAY & ACTIONS ---
        if st.session_state["kf_extracted_frames"] is not None:
            _display_extracted_frames(keyframe_algo_choice)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


def _extract_keyframes(frames, algo_choice, num_frames_target, video_name):
    """Extract keyframes using the selected algorithm."""
    st.session_state["kf_video_name"] = video_name

    if algo_choice == "Relative Quantization (Paper Implementation)":
        selected, idxs = uniform_sampling(frames, 5)
        processed = [draw_quantization_grid(f) for f in selected]
        st.session_state["kf_extracted_frames"] = processed
        st.session_state["kf_extracted_indices"] = idxs
        st.info("Visualizing Grid Encoding (Not saving raw frames for this mode).")
    elif algo_choice in ALGORITHM_MAP:
        algo_fn = ALGORITHM_MAP[algo_choice]
        selected, idxs = algo_fn(frames, num_frames_target)
        st.session_state["kf_extracted_frames"] = selected
        st.session_state["kf_extracted_indices"] = idxs
        st.success(f"Extracted {len(selected)} frames using {algo_choice}.")


def _display_extracted_frames(algo_choice):
    """Display extracted frames and save options."""
    st.divider()

    # --- SAVE FRAMES BUTTON ---
    st.text_input(
        "Output folder path for keyframes",
        key="kf_output_folder",
        placeholder="/absolute/path/to/output",
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
