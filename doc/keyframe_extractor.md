# Keyframe Extractor — Technical Documentation

## Overview

The Keyframe Extractor is a component of the WLASL (Word-Level American Sign Language) dataset pipeline. Its job is to take a raw sign language video and select the **most meaningful frames** from it — reducing hundreds of frames down to a fixed target count (default: 30). These selected frames, called **keyframes**, are the ones that best capture the motion and meaning of the sign being performed.

This is critical because:
- Raw video contains many redundant frames (e.g., the signer standing still before/after the sign).
- The downstream model expects a fixed number of frames per sample.
- Intelligent frame selection preserves the most information with the fewest frames.

---

## Source Files

| File | Purpose |
|------|---------|
| `keyframe_extractor/algorithms.py` | All 9 keyframe extraction algorithms |
| `keyframe_extractor/video_utils.py` | Video I/O (reading frames from video files) |
| `keyframe_extractor/app.py` | Streamlit UI that lets users choose an algorithm and extract keyframes |

---

## How Video Reading Works

Before any algorithm runs, the video must be read into memory as a list of individual image frames.

**File:** `video_utils.py`  
**Function:** `get_frames_from_video(video_path)`

1. Opens the video file using OpenCV's `cv2.VideoCapture`.
2. Reads frames one-by-one in a loop until the video ends.
3. Converts each frame from **BGR** (OpenCV's default) to **RGB** (standard for display/ML).
4. Returns a Python list of RGB numpy arrays, where each array has shape `(height, width, 3)`.

---

## Keyframe Extraction Algorithms

All algorithms live in `algorithms.py` and share the same interface:

```python
def algorithm_name(frames: list, target_count: int) -> tuple[list, list]:
    """
    Args:
        frames: List of RGB image frames (numpy arrays).
        target_count: How many keyframes to select (e.g., 30).

    Returns:
        selected_frames: List of selected frame images.
        selected_indices: List of original frame indices (for labeling).
    """
```

### 1. Uniform Sampling (Standard)

**How it works:**  
Picks frames at evenly spaced intervals across the entire video. If the video has 300 frames and you want 30, it picks every 10th frame.

**Algorithm:**
- Uses `np.linspace(0, total_frames - 1, target_count)` to compute equally spaced indices.
- Returns the frames at those positions.

**When to use:** Baseline approach. Works well when motion is evenly distributed throughout the video. Simple and fast.

**Limitations:** Misses action peaks if the signer pauses at the start/end. Doesn't adapt to content.

---

### 2. Motion Detection (Action Segments)

**How it works:**  
Selects frames where the most motion is happening by measuring the pixel-level difference between consecutive frames.

**Algorithm:**
1. Convert all frames to grayscale.
2. For each pair of consecutive frames, compute `cv2.absdiff()` (absolute pixel difference).
3. Sum all the pixel differences to get a **motion score** for each frame.
4. Sort by motion score (descending) and pick the top `target_count` frames.
5. Re-sort the selected frames by their original position to maintain temporal order.

**When to use:** Good for isolating the active part of the sign. Frames during fast hand movements get higher scores.

**Limitations:** Sensitive to camera shake or background motion. May miss slow but meaningful movements.

---

### 3. Optical Flow (Motion Magnitude)

**How it works:**  
Uses Farneback optical flow to measure the actual movement direction and speed of pixels between frames, rather than just pixel differences.

**Algorithm:**
1. Convert all frames to grayscale.
2. For each consecutive pair, compute **Farneback optical flow** using `cv2.calcOpticalFlowFarneback()`.
3. Convert the flow vectors (dx, dy) to polar coordinates to get **magnitude** (speed).
4. The mean magnitude across all pixels is the frame's motion score.
5. Pick the top `target_count` frames by score, then re-sort by position.

**When to use:** More robust than raw pixel differences. Better at detecting smooth, continuous motion.

**Limitations:** Computationally heavier than simple motion detection.

---

### 4. Farneback Dense Optical Flow

**How it works:**  
Same as Optical Flow, but with an additional **Gaussian blur** preprocessing step to reduce noise.

**Algorithm:**
1. Convert frames to grayscale.
2. Apply `cv2.GaussianBlur(gray, (5, 5), 0)` to each frame before computing flow.
3. Compute Farneback optical flow on the blurred frames.
4. Score and select as before.

**When to use:** When the video has noise or compression artifacts that would confuse raw optical flow.

---

### 5. Keypoint/Skeleton (MediaPipe Pose+Hands)

**How it works:**  
Uses MediaPipe's **PoseLandmarker** and **HandLandmarker** to detect body keypoints (skeleton joints and hand landmarks) in each frame, then measures how much these keypoints move between frames.

**Algorithm:**
1. Run MediaPipe PoseLandmarker (33 body landmarks) and HandLandmarker (21 landmarks per hand, up to 2 hands) on each frame.
2. For each pair of consecutive frames, compute the Euclidean distance between corresponding keypoints.
3. The average keypoint displacement is the frame's motion score.
4. Pick the top `target_count` frames.

**When to use:** Best for sign language specifically, because it measures the movement of the signer's body and hands rather than pixels. Ignores background movement entirely.

**Limitations:** Slowest algorithm due to running pose+hand detection on every frame. Requires model file downloads on first use.

**Model files:** Automatically downloaded to `.models/` directory on first use:
- `pose_landmarker_lite.task` (~4 MB)
- `hand_landmarker.task` (~6 MB)

---

### 6. CNN + LSTM (Feature Change)

**How it works:**  
Extracts visual features from each frame using CNN-like operations (edge detection + downsampling), then runs them through a lightweight LSTM to measure temporal changes.

**Algorithm:**
1. **Feature extraction** (per frame):
   - Convert to grayscale and resize to 64×64.
   - Compute Sobel edges (horizontal + vertical) → magnitude.
   - Compute Laplacian (second-derivative edges).
   - Downsample each feature map to 8×8 and flatten → 192-dimensional feature vector.
   - Normalize features across all frames (zero mean, unit variance).

2. **LSTM scoring:**
   - Initialize a single-layer LSTM with fixed random weights (seed=42, so deterministic).
   - Feed features through the LSTM one frame at a time.
   - The "score" for each frame is the L2 norm of the hidden state change: `||h_new - h_old||`.
   - Frames with large hidden state changes are considered "surprising" or "significant".

3. Pick top `target_count` frames by score.

**When to use:** Captures temporal dynamics better than per-frame methods. Good at detecting transitions and significant visual changes.

**Note:** This uses fixed (random) weights, so it's not a trained model — it's a heuristic that leverages the LSTM's recurrent structure to detect temporal novelty.

---

### 7. Transformer Attention (Self-Attention)

**How it works:**  
Computes self-attention scores across all frames to identify which frames are most "attended to" by other frames — a proxy for importance.

**Algorithm:**
1. Extract the same 192-dimensional features as the CNN+LSTM method.
2. Normalize features.
3. Project features into Query (Q), Key (K), and Value (V) matrices using fixed random weight matrices.
4. Compute self-attention: `softmax(Q × K^T / sqrt(d_k))`.
5. The **importance score** for each frame is the average attention it receives from all other frames.
6. Pick the top `target_count` frames.

**When to use:** Identifies globally important frames — frames that are "representative" of the overall video content.

**Note:** Like the CNN+LSTM, this uses fixed random projections, not trained weights.

---

### 8. Voxel Spatio-Temporal (Motion Volume)

**How it works:**  
Treats the video as a 3D volume (width × height × time) and measures the fraction of spatial cells where motion occurs.

**Algorithm:**
1. Resize each frame to a grid (default: 32×32) and convert to grayscale.
2. For each consecutive pair, compute the absolute difference.
3. Threshold the difference (default: 12) to create a binary "motion voxel" (1 = motion, 0 = no motion).
4. The frame's score is the mean of its motion voxel (fraction of cells with motion).
5. Pick the top `target_count` frames.

**When to use:** Less sensitive to small, noisy movements. Good at detecting frames with large, broad motions.

**Parameters:**
- `grid_size` (default: 32) — Resolution of the spatial grid. Larger = more detail.
- `threshold` (default: 12) — Pixel difference below this is ignored. Higher = only big motion.

---

### 9. Relative Quantization (Paper Implementation)

**How it works:**  
This is a **visualization mode**, not a frame selection algorithm for training. It uniformly samples 5 frames and overlays a spatial grid on each one, showing how hand positions could be encoded relative to a grid coordinate system.

**Algorithm:**
1. Uniformly sample 5 frames from the video.
2. Draw a 10×10 green grid centered on each frame using `draw_quantization_grid()`.
3. Mark the center point with a red dot.

**When to use:** For visualization and understanding the paper's approach to spatial encoding. Not for generating training data.

---

## Algorithm Registry

The file provides an `ALGORITHM_MAP` dictionary and an `ALGORITHM_NAMES` list for clean UI dispatch:

```python
from algorithms import ALGORITHM_MAP, ALGORITHM_NAMES

# Get algorithm function by name
algo_fn = ALGORITHM_MAP["Uniform Sampling (Standard)"]
selected_frames, indices = algo_fn(frames, target_count=30)
```

---

## How to Choose an Algorithm

| Scenario | Recommended Algorithm |
|---|---|
| Quick extraction, any video | Uniform Sampling |
| Sign language, fast motion | Motion Detection or Optical Flow |
| Sign language, noisy video | Farneback Dense Optical Flow |
| Sign language, best quality | Keypoint/Skeleton (MediaPipe) |
| General purpose, good balance | CNN + LSTM |
| Finding representative frames | Transformer Attention |
| Large/broad motions only | Voxel Spatio-Temporal |
