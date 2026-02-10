"""
WLASL Keyframe & Feature Extractor — Streamlit App

This is the main UI layer. All business logic is delegated to:
  - video_utils.py       → Video I/O
  - algorithms.py        → Keyframe extraction algorithms
  - file_utils.py        → Frame saving
  - landmark_extractor.py → MediaPipe landmark extraction
  - data_exporter.py     → Numpy export
"""

import tempfile
import os

import numpy as np
import streamlit as st

from video_utils import get_frames_from_video
from algorithms import ALGORITHM_MAP, ALGORITHM_NAMES, uniform_sampling, draw_quantization_grid
from file_utils import save_frames_to_folder
from landmark_extractor import (
    LANDMARK_METHODS,
    extract_landmarks_from_frames,
    draw_landmarks_on_frames,
    TOTAL_FEATURES,
)
from data_exporter import landmarks_to_numpy, save_landmarks_as_npy


# --- PAGE CONFIG ---
st.set_page_config(page_title="WLASL Keyframe Extractor", layout="wide")
st.title("🖐️ WLASL Keyframe & Feature Extractor")

# --- MAIN CONTROLS ---
st.subheader("Keyframe Extractor")
keyframe_algo_choice = st.selectbox("Keyframe Extraction Algorithm", ALGORITHM_NAMES)

st.subheader("Landmark Extractor")
landmark_choice = st.selectbox("Landmark Extraction Method", LANDMARK_METHODS)

# Normalization & Localization checkboxes (only relevant when landmarks are enabled)
col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    normalize_landmarks = st.checkbox(
        "Normalize Landmarks",
        value=False,
        disabled=(landmark_choice == "None"),
        help="Scale landmarks to [0, 1] range to remove body-size variance.",
    )
with col_opt2:
    localize_landmarks = st.checkbox(
        "Localize Landmarks",
        value=False,
        disabled=(landmark_choice == "None"),
        help="Translate landmarks so body center (mid-hip) is at the origin.",
    )

num_frames_target = st.slider("Target Number of Frames", 10, 50, 30)

# --- SESSION STATE ---
if "extracted_frames" not in st.session_state:
    st.session_state["extracted_frames"] = None
if "extracted_indices" not in st.session_state:
    st.session_state["extracted_indices"] = None
if "video_name" not in st.session_state:
    st.session_state["video_name"] = ""
if "extracted_landmarks" not in st.session_state:
    st.session_state["extracted_landmarks"] = None
if "raw_landmarks" not in st.session_state:
    st.session_state["raw_landmarks"] = None

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload a Sign Language Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name

    try:
        frames = get_frames_from_video(video_path)

        if frames:
            st.subheader(f"Original Video ({len(frames)} frames)")
            st.image(frames[len(frames) // 2], caption="Middle Frame Preview", width=400)

            # --- EXTRACTION BUTTON ---
            if st.button("Extract Frames"):
                st.session_state["video_name"] = uploaded_file.name
                st.session_state["extracted_landmarks"] = None  # Reset landmarks
                st.session_state["raw_landmarks"] = None

                if keyframe_algo_choice == "Relative Quantization (Paper Implementation)":
                    selected, idxs = uniform_sampling(frames, 5)
                    processed = [draw_quantization_grid(f) for f in selected]
                    st.session_state["extracted_frames"] = processed
                    st.session_state["extracted_indices"] = idxs
                    st.info(
                        "Visualizing Grid Encoding (Not saving raw frames for this mode)."
                    )
                elif keyframe_algo_choice in ALGORITHM_MAP:
                    algo_fn = ALGORITHM_MAP[keyframe_algo_choice]
                    selected, idxs = algo_fn(frames, num_frames_target)
                    st.session_state["extracted_frames"] = selected
                    st.session_state["extracted_indices"] = idxs
                    st.success(
                        f"Extracted {len(selected)} frames using {keyframe_algo_choice}."
                    )

            # --- DISPLAY & ACTIONS ---
            if st.session_state["extracted_frames"] is not None:
                st.divider()

                # --- SAVE FRAMES BUTTON ---
                if st.button("💾 Save Frames to Folder"):
                    if keyframe_algo_choice == "Relative Quantization (Paper Implementation)":
                        st.warning(
                            "The Paper Implementation mode is for visualization only. "
                            "Switch to Uniform or Motion to save raw data for training."
                        )
                    else:
                        success, msg = save_frames_to_folder(
                            st.session_state["extracted_frames"],
                            st.session_state["extracted_indices"],
                            st.session_state["video_name"],
                        )
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)

                # --- LANDMARK EXTRACTION ---
                if landmark_choice != "None":
                    st.divider()
                    st.subheader("🦴 Landmark Extraction")

                    if st.button("🔍 Extract Landmarks"):
                        with st.spinner("Extracting landmarks with MediaPipe..."):
                            # Always extract raw landmarks first (for visualization)
                            raw = extract_landmarks_from_frames(
                                st.session_state["extracted_frames"],
                                method=landmark_choice,
                                normalize=False,
                                localize=False,
                            )
                            raw = landmarks_to_numpy(raw, num_frames_target)
                            st.session_state["raw_landmarks"] = raw

                            # Apply optional transformations for data export
                            if normalize_landmarks or localize_landmarks:
                                transformed = extract_landmarks_from_frames(
                                    st.session_state["extracted_frames"],
                                    method=landmark_choice,
                                    normalize=normalize_landmarks,
                                    localize=localize_landmarks,
                                )
                                transformed = landmarks_to_numpy(transformed, num_frames_target)
                            else:
                                transformed = raw

                            st.session_state["extracted_landmarks"] = transformed
                            st.success(
                                f"Extracted landmarks: shape {transformed.shape} "
                                f"({transformed.shape[0]} frames × {transformed.shape[1]} features)"
                            )

                    # Show landmark data summary & save button
                    if st.session_state["extracted_landmarks"] is not None:
                        landmarks_data = st.session_state["extracted_landmarks"]

                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Shape", f"{landmarks_data.shape}")
                        with col2:
                            non_zero_frames = int(
                                np.sum(np.any(landmarks_data != 0, axis=1))
                            )
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
                                width='stretch',
                                height=300,
                            )

                        # Save as .npy
                        if st.button("💾 Save Landmarks as .npy"):
                            success, msg = save_landmarks_as_npy(
                                landmarks_data,
                                st.session_state["video_name"],
                            )
                            if success:
                                st.success(msg)
                            else:
                                st.error(msg)

                        # --- LANDMARK VISUALIZATION ---
                        st.divider()
                        st.subheader("🎨 Landmark Visualization")
                        st.caption(
                            "🟢 Pose skeleton  ·  🟠 Left hand  ·  🔵 Right hand"
                        )

                        # Use RAW landmarks (no normalization/localization) for visualization
                        raw_lm = st.session_state["raw_landmarks"]
                        viz_frames = draw_landmarks_on_frames(
                            st.session_state["extracted_frames"],
                            raw_lm,
                            method=landmark_choice,
                        )

                        viz_cols = st.columns(5)
                        for i, vf in enumerate(viz_frames):
                            with viz_cols[i % 5]:
                                st.image(
                                    vf,
                                    caption=f"Frame {st.session_state['extracted_indices'][i]}",
                                    width='stretch',
                                )

                # --- PREVIEW SECTION ---
                st.divider()
                st.write("**Extracted Keyframes Preview:**")
                cols = st.columns(5)
                for i, frame in enumerate(st.session_state["extracted_frames"]):
                    with cols[i % 5]:
                        st.image(
                            frame,
                            caption=f"Frame {st.session_state['extracted_indices'][i]}",
                            width='stretch',
                        )
        else:
            st.error("Could not extract frames.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)