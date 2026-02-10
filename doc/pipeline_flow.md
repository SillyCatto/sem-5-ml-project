# Pipeline Flow — End-to-End Documentation

## Overview

This document explains how the **Keyframe Extractor** and **Landmark Extractor** work together as a complete pipeline to transform a raw sign language video into a machine-learning-ready `.npy` file.

The pipeline is designed for **WLASL (Word-Level American Sign Language) recognition** — it produces training data in the exact format expected by the recognition model.

---

## Pipeline Architecture

```
┌───────────┐     ┌────────────────┐     ┌─────────────────┐     ┌──────────┐
│   Video   │────▶│   Keyframe     │────▶│   Landmark      │────▶│  .npy    │
│   (.mp4)  │     │   Extraction   │     │   Extraction    │     │  File    │
│           │     │  (30 frames)   │     │  (258 features) │     │(30, 258) │
└───────────┘     └────────────────┘     └─────────────────┘     └──────────┘
                   ▲                       ▲
                   │                       │
              Choose from 9           Choose: Pose,
              algorithms              Hands, or Both
                                      + Normalize?
                                      + Localize?
```

---

## Step-by-Step Flow

### Step 1: Video Upload

The user uploads a sign language video file (`.mp4`, `.mov`, or `.avi`) through the Streamlit web interface.

**What happens internally:**
1. The uploaded file is written to a temporary file on disk (required by OpenCV).
2. `video_utils.get_frames_from_video()` reads all frames from the video.
3. Each frame is converted from BGR (OpenCV's format) to RGB.
4. A preview of the middle frame is displayed in the UI.

**Output:** A Python list of RGB numpy arrays, one per frame (e.g., 150 frames of shape `(480, 640, 3)`).

---

### Step 2: Keyframe Extraction

The user selects one of 9 algorithms and clicks **"Extract Frames"**.

**What happens internally:**
1. The selected algorithm receives the full frame list and the target count (default: 30).
2. The algorithm analyzes the frames (motion, optical flow, keypoints, etc.) and selects the most informative ones.
3. The selected frames and their original indices are stored in session state.

**Output:** Two lists:
- `extracted_frames` — 30 RGB images (the keyframes)
- `extracted_indices` — 30 integers (which frames from the original video were selected)

**Displayed in UI:** A grid of keyframe thumbnails labeled with their original frame numbers.

See [Keyframe Extractor Documentation](keyframe_extractor.md) for algorithm details.

---

### Step 3: Landmark Extraction

The user selects a landmark method, optionally enables Normalization/Localization, and clicks **"🔍 Extract Landmarks"**.

**What happens internally:**
1. MediaPipe model files are downloaded (first time only) to `.models/` directory.
2. **Raw extraction pass:** Each keyframe is processed by MediaPipe PoseLandmarker and/or HandLandmarker. Per frame:
   - 33 pose landmarks × 4 values = 132 floats
   - 21 left hand landmarks × 3 values = 63 floats
   - 21 right hand landmarks × 3 values = 63 floats
   - Missing landmarks (e.g., hand not visible) are filled with zeros.
3. The raw landmarks are padded/truncated to exactly the target frame count (30 by default).
4. Raw landmarks are saved in session state for **visualization**.
5. **Transformation pass** (if normalization or localization is enabled):
   - Landmarks are extracted again.
   - Localization centers all coordinates relative to the mid-hip point.
   - Normalization scales all values to [0, 1] range.
   - Transformed landmarks are saved separately for **data export**.

**Output:** Two numpy arrays, both of shape `(30, 258)`:
- `raw_landmarks` — Untransformed, used for visualization.
- `extracted_landmarks` — Optionally normalized/localized, used for data preview and `.npy` export.

**Displayed in UI:**
- Summary metrics (shape, non-zero frame count, value range).
- Expandable data table showing the full 30×258 matrix.

See [Landmark Extractor Documentation](landmark_extractor.md) for feature layout and transformation details.

---

### Step 4: Landmark Visualization

After extraction, the landmarks are automatically drawn on the keyframe images.

**What happens internally:**
1. For each keyframe, the corresponding row from `raw_landmarks` is reshaped:
   - Pose: 33 landmarks with (x, y, z, visibility)
   - Left hand: 21 landmarks with (x, y, z)
   - Right hand: 21 landmarks with (x, y, z)
2. The (x, y) coordinates (in [0, 1] range) are scaled to pixel coordinates.
3. Skeleton connections and dots are drawn on a copy of the frame using OpenCV.

**Color coding:**
- 🟢 **Green** — Pose skeleton (body joints and connections)
- 🟠 **Orange** — Left hand landmarks and finger connections
- 🔵 **Blue** — Right hand landmarks and finger connections

**Why raw landmarks are used for visualization:**  
Normalized/localized coordinates don't map to valid pixel positions (they can be negative or re-scaled), so the visualization always uses the original coordinates.

---

### Step 5: Save as .npy

The user clicks **"💾 Save Landmarks as .npy"**.

**What happens internally:**
1. A folder picker dialog opens (tkinter).
2. The `extracted_landmarks` array (shape `(30, 258)`, with any normalization/localization applied) is saved as `{video_name}.npy`.

**Output:** A single `.npy` file that can be loaded directly for model training:
```python
import numpy as np
data = np.load("video_name.npy")
print(data.shape)  # (30, 258)
```

---

### Optional: Save Frames as Images

At any point after keyframe extraction, the user can also click **"💾 Save Frames to Folder"** to save the raw keyframe images as `.jpg` files for manual inspection or other uses.

---

## Data Flow Diagram

```
Video File (.mp4)
    │
    ▼
┌─────────────────────────────────┐
│ video_utils.get_frames_from_video │
│ ─ OpenCV reads all frames         │
│ ─ BGR → RGB conversion            │
└─────────────┬───────────────────┘
              │ List of RGB frames
              ▼
┌─────────────────────────────────┐
│ algorithms.py (chosen algorithm)  │
│ ─ Analyzes motion/features        │
│ ─ Selects top N frames            │
└─────────────┬───────────────────┘
              │ 30 keyframes + indices
              ▼
┌─────────────────────────────────┐
│ landmark_extractor.py             │
│ ─ MediaPipe Pose + Hands          │
│ ─ 258 features per frame          │
│ ─ Optional: normalize/localize    │
└──────┬──────────────┬───────────┘
       │              │
  raw landmarks   transformed landmarks
       │              │
       ▼              ▼
┌───────────┐  ┌──────────────┐
│ Visualize │  │ Data Preview │
│ on frames │  │ & .npy Save  │
└───────────┘  └──────────────┘
```

---

## File Structure

```
sem-5-ml-project/
├── keyframe_extractor/
│   ├── app.py                  ← Streamlit UI (thin orchestration layer)
│   ├── video_utils.py          ← Video I/O
│   ├── algorithms.py           ← 9 keyframe extraction algorithms
│   ├── file_utils.py           ← Frame saving (tkinter folder picker)
│   ├── landmark_extractor.py   ← Landmark extraction + visualization
│   ├── data_exporter.py        ← Numpy conversion + .npy saving
│   └── .models/                ← Auto-downloaded MediaPipe model files
├── model/
│   └── WLASL_recognition_using_Action_Detection.ipynb  ← Training notebook
├── doc/
│   ├── keyframe_extractor.md   ← This documentation
│   ├── landmark_extractor.md
│   └── pipeline_flow.md
├── main.py
├── pyproject.toml
└── uv.lock
```

---

## How to Run

```bash
cd h:\Projects\sem-5-ml-project
uv run streamlit run keyframe_extractor/app.py
```

This starts a local web server (default: http://localhost:8501) with the full pipeline UI.

---

## Configuration & Dependencies

**Package manager:** `uv` (with `pyproject.toml`)

**Dependencies:**
| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.54.0 | Web UI framework |
| `opencv-python` | ≥4.13.0.92 | Video I/O, image processing, optical flow |
| `numpy` | ≥2.4.2 | Array operations, `.npy` saving |
| `mediapipe` | ≥0.10.32 | Pose and hand landmark detection |

**Runtime requirements:**
- Python ≥3.13
- Internet connection (first run only, to download MediaPipe model files)
- Webcam not required (works with video files only)

---

## Relationship to the Model Notebook

The recognition model ([WLASL_recognition_using_Action_Detection.ipynb](../model/WLASL_recognition_using_Action_Detection.ipynb)) expects:

1. **Input:** `.npy` files with shape `(30, 258)` — exactly what this pipeline produces.
2. **Directory structure:** One folder per word/class, containing multiple `.npy` files (one per video).
3. **Feature layout:** `[Pose_132 | Left_Hand_63 | Right_Hand_63]` — matches our extraction format.

The notebook handles data augmentation, model training (LSTM-based), and evaluation. This pipeline is the **data preparation step** that comes before training.

```
This Pipeline                          Model Notebook
─────────────                          ──────────────
Video → Keyframes → Landmarks → .npy   →  Load .npy → Train LSTM → Predict
```
