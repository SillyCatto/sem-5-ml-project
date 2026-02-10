import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import tkinter as tk
from tkinter import filedialog

# --- PAGE CONFIG ---
st.set_page_config(page_title="WLASL Keyframe Extractor", layout="wide")

st.title("🖐️ WLASL Keyframe & Feature Extractor")

# --- SIDEBAR ---
st.sidebar.header("Settings")
algo_choice = st.sidebar.selectbox(
    "Choose Extraction Algorithm",
    (
        "Uniform Sampling (Standard)",
        "Motion Detection (Action Segments)",
        "Optical Flow (Motion Magnitude)",
        "Relative Quantization (Paper Implementation)"
    )
)

num_frames_target = st.sidebar.slider("Target Number of Frames", 10, 50, 30)

# --- HELPER FUNCTIONS ---

def get_frames_from_video(video_path):
    """Reads all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def uniform_sampling(frames, target_count):
    total_frames = len(frames)
    if total_frames == 0:
        return [], []
    indices = np.linspace(0, total_frames - 1, target_count, dtype=int)
    selected_frames = [frames[i] for i in indices]
    return selected_frames, indices

def motion_based_extraction(frames, target_count):
    if len(frames) < 2:
        return frames, list(range(len(frames)))
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    motion_scores = []
    for i in range(len(gray_frames) - 1):
        score = np.sum(cv2.absdiff(gray_frames[i], gray_frames[i+1]))
        motion_scores.append(score)

    # Pad the last frame score to match length
    motion_scores.append(0)
    motion_scores = np.array(motion_scores)

    top_indices = np.argsort(motion_scores)[::-1][:target_count]
    top_indices = np.sort(top_indices)
    selected_frames = [frames[i] for i in top_indices]
    return selected_frames, top_indices

def optical_flow_extraction(frames, target_count):
    if len(frames) < 2:
        return frames, list(range(len(frames)))

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flow_scores = []

    for i in range(len(gray_frames) - 1):
        prev = gray_frames[i]
        nxt = gray_frames[i + 1]
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_scores.append(float(np.mean(mag)))

    # Pad the last frame score to match length
    flow_scores.append(0)
    flow_scores = np.array(flow_scores)

    top_indices = np.argsort(flow_scores)[::-1][:target_count]
    top_indices = np.sort(top_indices)
    selected_frames = [frames[i] for i in top_indices]
    return selected_frames, top_indices

def draw_quantization_grid(frame):
    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2
    overlay = frame.copy()
    step_x = w // 10
    step_y = h // 10
    color = (0, 255, 0)
    for i in range(-5, 6):
        cv2.line(overlay, (center_x + i*step_x, 0), (center_x + i*step_x, h), color, 1)
        cv2.line(overlay, (0, center_y + i*step_y), (w, center_y + i*step_y), color, 1)
    cv2.circle(overlay, (center_x, center_y), 5, (255, 0, 0), -1)
    return cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

def save_frames_to_folder(frames, indices, video_name):
    # 1. Open Folder Picker
    root = tk.Tk()
    root.withdraw() # Hide the main window
    root.wm_attributes('-topmost', 1) # Bring picker to front
    target_dir = filedialog.askdirectory(title="Select Folder to Save Frames")
    root.destroy()

    if not target_dir:
        return False, "No folder selected."

    # 2. Create Sub-folder
    video_clean_name = os.path.splitext(video_name)[0]
    save_path = os.path.join(target_dir, video_clean_name)

    try:
        os.makedirs(save_path, exist_ok=True)
    except OSError as e:
        return False, f"Error creating directory: {e}"

    # 3. Save Images
    count = 0
    for frame, idx in zip(frames, indices):
        # Convert RGB back to BGR for OpenCV saving
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = f"{video_clean_name}-frame-{idx}.jpg"
        full_path = os.path.join(save_path, filename)
        cv2.imwrite(full_path, bgr_frame)
        count += 1

    return True, f"Successfully saved {count} frames to: {save_path}"

# --- MAIN UI LOGIC ---

# Initialize Session State to hold data across button clicks
if 'extracted_frames' not in st.session_state:
    st.session_state['extracted_frames'] = None
if 'extracted_indices' not in st.session_state:
    st.session_state['extracted_indices'] = None
if 'video_name' not in st.session_state:
    st.session_state['video_name'] = ""

uploaded_file = st.file_uploader("Upload a Sign Language Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name

    try:
        frames = get_frames_from_video(video_path)

        if frames:
            st.subheader(f"Original Video ({len(frames)} frames)")
            st.image(frames[len(frames)//2], caption="Middle Frame Preview", width=400)

            # EXTRACTION BUTTON
            if st.button("Extract Frames"):
                st.session_state['video_name'] = uploaded_file.name

                if algo_choice == "Uniform Sampling (Standard)":
                    selected, idxs = uniform_sampling(frames, num_frames_target)
                    st.session_state['extracted_frames'] = selected
                    st.session_state['extracted_indices'] = idxs
                    st.success(f"Extracted {len(selected)} frames using Uniform Sampling.")

                elif algo_choice == "Motion Detection (Action Segments)":
                    selected, idxs = motion_based_extraction(frames, num_frames_target)
                    st.session_state['extracted_frames'] = selected
                    st.session_state['extracted_indices'] = idxs
                    st.success(f"Extracted {len(selected)} frames using Motion Detection.")

                elif algo_choice == "Optical Flow (Motion Magnitude)":
                    selected, idxs = optical_flow_extraction(frames, num_frames_target)
                    st.session_state['extracted_frames'] = selected
                    st.session_state['extracted_indices'] = idxs
                    st.success(f"Extracted {len(selected)} frames using Optical Flow.")

                elif algo_choice == "Relative Quantization (Paper Implementation)":
                    selected, idxs = uniform_sampling(frames, 5)
                    processed = [draw_quantization_grid(f) for f in selected]
                    st.session_state['extracted_frames'] = processed
                    st.session_state['extracted_indices'] = idxs
                    st.info("Visualizing Grid Encoding (Not saving raw frames for this mode).")

            # DISPLAY & SAVE SECTION
            if st.session_state['extracted_frames'] is not None:
                st.divider()

                # SAVE BUTTON
                if st.button("💾 Save Frames to Folder"):
                    if algo_choice == "Relative Quantization (Paper Implementation)":
                         st.warning("The Paper Implementation mode is for visualization only. Switch to Uniform or Motion to save raw data for training.")
                    else:
                        success, msg = save_frames_to_folder(
                            st.session_state['extracted_frames'],
                            st.session_state['extracted_indices'],
                            st.session_state['video_name']
                        )
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)

                # --- PREVIEW SECTION ---
                st.write("**Extracted Keyframes Preview:**")

                cols = st.columns(5)
                for i, frame in enumerate(st.session_state['extracted_frames']):
                    with cols[i % 5]:
                        st.image(frame, caption=f"Frame {st.session_state['extracted_indices'][i]}", use_container_width=True)

        else:
            st.error("Could not extract frames.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)