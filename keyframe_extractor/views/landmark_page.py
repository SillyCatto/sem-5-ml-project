"""
Landmark Extractor Page

Convert extracted keyframes to MediaPipe landmarks for model training.
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from landmark_extractor import (
    LANDMARK_METHODS,
    extract_landmarks_from_frames,
    draw_landmarks_on_frames,
)
from data_exporter import landmarks_to_numpy, save_landmarks_as_npy


def init_session_state():
    """Initialize session state variables for landmark extractor."""
    if "lm_loaded_frames" not in st.session_state:
        st.session_state["lm_loaded_frames"] = None
    if "lm_frame_indices" not in st.session_state:
        st.session_state["lm_frame_indices"] = None
    if "lm_folder_name" not in st.session_state:
        st.session_state["lm_folder_name"] = ""
    if "lm_extracted_landmarks" not in st.session_state:
        st.session_state["lm_extracted_landmarks"] = None
    if "lm_raw_landmarks" not in st.session_state:
        st.session_state["lm_raw_landmarks"] = None
    if "lm_selected_folder" not in st.session_state:
        st.session_state["lm_selected_folder"] = None
    if "lm_save_folder" not in st.session_state:
        st.session_state["lm_save_folder"] = ""


def render():
    """Render the landmark extractor page."""
    init_session_state()

    st.caption("Convert extracted keyframes to MediaPipe landmarks for model training.")

    # --- CONTROLS ---
    landmark_methods = [m for m in LANDMARK_METHODS if m != "None"]
    landmark_choice = st.selectbox("Landmark Extraction Method", landmark_methods)

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        normalize_landmarks = st.checkbox(
            "Normalize Landmarks",
            value=False,
            help="Scale landmarks to [0, 1] range to remove body-size variance.",
        )
    with col_opt2:
        localize_landmarks = st.checkbox(
            "Localize Landmarks",
            value=False,
            help="Translate landmarks so mid-chest (between shoulders) is at the origin.",
        )

    num_frames_target = st.slider(
        "Target Number of Frames", 10, 50, 30, key="lm_num_frames"
    )

    st.divider()

    # --- LOAD KEYFRAMES FROM FOLDER ---
    _render_folder_loader()

    # --- DISPLAY LOADED FRAMES & EXTRACT LANDMARKS ---
    if st.session_state["lm_loaded_frames"] is not None:
        _render_loaded_frames()
        _render_landmark_extraction(
            landmark_choice, normalize_landmarks, localize_landmarks, num_frames_target
        )

        if st.session_state["lm_extracted_landmarks"] is not None:
            _render_landmark_results(landmark_choice)


def _render_folder_loader():
    """Render the folder selection UI."""
    st.subheader("📂 Load Keyframes")
    st.caption("Select a folder containing extracted keyframe images (.png or .jpg).")

    folder_path = st.text_input(
        "Keyframes folder path",
        value=st.session_state.get("lm_selected_folder") or "",
        placeholder="/absolute/path/to/keyframes",
    )

    if folder_path:
        st.session_state["lm_selected_folder"] = folder_path
        if st.button("📥 Load Keyframes"):
            _load_keyframes_from_folder()
    else:
        st.info("Paste the folder path that contains the extracted keyframes.")


def _load_keyframes_from_folder():
    """Load keyframe images from the selected folder."""
    folder_path = Path(st.session_state["lm_selected_folder"])
    image_files = sorted(folder_path.glob("*.png")) + sorted(folder_path.glob("*.jpg"))

    if not image_files:
        st.error("No image files found in the selected folder.")
        return

    frames = []
    indices = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)
            # Extract frame index from filename (e.g., "video-frame-42.jpg" -> 42)
            try:
                idx = int(img_path.stem.split("-")[-1])
            except ValueError:
                idx = len(indices)
            indices.append(idx)

    st.session_state["lm_loaded_frames"] = frames
    st.session_state["lm_frame_indices"] = indices
    st.session_state["lm_folder_name"] = folder_path.name
    st.session_state["lm_extracted_landmarks"] = None
    st.session_state["lm_raw_landmarks"] = None
    st.success(f"Loaded {len(frames)} keyframes from '{folder_path.name}'.")


def _render_loaded_frames():
    """Display the loaded keyframes."""
    st.divider()
    st.subheader("🖼️ Loaded Keyframes")

    cols = st.columns(5)
    for i, frame in enumerate(st.session_state["lm_loaded_frames"]):
        with cols[i % 5]:
            st.image(
                frame,
                caption=f"Frame {st.session_state['lm_frame_indices'][i]}",
                use_container_width=True,
            )


def _render_landmark_extraction(
    landmark_choice, normalize, localize, num_frames_target
):
    """Render the landmark extraction UI and handle extraction."""
    st.divider()
    st.subheader("🔍 Extract Landmarks")

    if st.button("🔍 Extract Landmarks from Keyframes"):
        with st.spinner("Extracting landmarks with MediaPipe..."):
            # Always extract raw landmarks first (for visualization)
            raw = extract_landmarks_from_frames(
                st.session_state["lm_loaded_frames"],
                method=landmark_choice,
                normalize=False,
                localize=False,
            )
            raw = landmarks_to_numpy(raw, num_frames_target)
            st.session_state["lm_raw_landmarks"] = raw

            # Apply optional transformations for data export
            if normalize or localize:
                transformed = extract_landmarks_from_frames(
                    st.session_state["lm_loaded_frames"],
                    method=landmark_choice,
                    normalize=normalize,
                    localize=localize,
                )
                transformed = landmarks_to_numpy(transformed, num_frames_target)
            else:
                transformed = raw

            st.session_state["lm_extracted_landmarks"] = transformed
            st.success(
                f"Extracted landmarks: shape {transformed.shape} "
                f"({transformed.shape[0]} frames × {transformed.shape[1]} features)"
            )


def _render_landmark_results(landmark_choice):
    """Render landmark results, stats, and visualization."""
    landmarks_data = st.session_state["lm_extracted_landmarks"]

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Shape", f"{landmarks_data.shape}")
    with col2:
        non_zero_frames = int(np.sum(np.any(landmarks_data != 0, axis=1)))
        st.metric("Non-zero Frames", f"{non_zero_frames}/{landmarks_data.shape[0]}")
    with col3:
        st.metric(
            "Value Range",
            f"[{landmarks_data.min():.3f}, {landmarks_data.max():.3f}]",
        )

    # Preview data
    with st.expander("📊 Preview Landmark Data"):
        st.dataframe(
            landmarks_data,
            use_container_width=True,
            height=300,
        )

    # Save as .npy
    st.text_input(
        "Output folder path for .npy",
        key="lm_save_folder",
        placeholder="/absolute/path/to/output",
    )

    if st.button("💾 Save Landmarks as .npy"):
        success, msg = save_landmarks_as_npy(
            landmarks_data,
            st.session_state["lm_folder_name"],
            target_dir=st.session_state.get("lm_save_folder") or None,
        )
        if success:
            st.success(msg)
        else:
            st.error(msg)

    # --- LANDMARK VISUALIZATION ---
    st.divider()
    st.subheader("🎨 Landmark Visualization")
    st.caption("🟢 Pose skeleton  ·  🟠 Left hand  ·  🔵 Right hand")

    # Use RAW landmarks (no normalization/localization) for visualization
    raw_lm = st.session_state["lm_raw_landmarks"]
    viz_frames = draw_landmarks_on_frames(
        st.session_state["lm_loaded_frames"],
        raw_lm,
        method=landmark_choice,
    )

    viz_cols = st.columns(5)
    for i, vf in enumerate(viz_frames):
        with viz_cols[i % 5]:
            st.image(
                vf,
                caption=f"Frame {st.session_state['lm_frame_indices'][i]}",
                use_container_width=True,
            )
