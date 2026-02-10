# Landmark Extractor — Technical Documentation

## Overview

The Landmark Extractor takes the keyframes selected by the Keyframe Extractor and detects **human body pose and hand landmarks** on each frame using Google's **MediaPipe** machine learning models. Each frame produces a flat array of **258 numerical values** representing the precise positions of body joints and hand keypoints. This data is what the sign language recognition model actually trains on — not raw pixel images.

---

## Source Files

| File | Purpose |
|------|---------|
| `keyframe_extractor/landmark_extractor.py` | Landmark extraction, normalization, localization, and visualization |
| `keyframe_extractor/data_exporter.py` | Numpy array conversion and `.npy` file saving |

---

## What Are Landmarks?

A **landmark** is a specific anatomical point on the human body that MediaPipe can detect and locate in an image. Each landmark has coordinates:

- **Pose landmarks** have 4 values: `(x, y, z, visibility)`
  - `x, y` — Position in the image (normalized to [0, 1] where (0,0) is top-left)
  - `z` — Depth relative to the hip midpoint (smaller = closer to camera)
  - `visibility` — Confidence that the landmark is visible (0 to 1)

- **Hand landmarks** have 3 values: `(x, y, z)`
  - Same as pose, but without visibility (hand models don't provide it)

---

## Feature Vector Layout

Each frame produces a flat array of **258 values**, structured as:

```
┌──────────────────────────────────────────────────────────┐
│ Pose Landmarks      │ Left Hand          │ Right Hand     │
│ 33 × 4 = 132 values │ 21 × 3 = 63 values │ 21 × 3 = 63   │
│ [0..131]             │ [132..194]          │ [195..257]     │
└──────────────────────────────────────────────────────────┘
Total: 132 + 63 + 63 = 258 features per frame
```

This matches the shape expected by the WLASL recognition model notebook: `(30, 258)` — 30 frames, 258 features each.

### Pose Landmarks (33 Points)

MediaPipe's PoseLandmarker detects these 33 body landmarks:

| Index | Landmark | Index | Landmark |
|-------|----------|-------|----------|
| 0 | Nose | 17 | Left pinky (pose) |
| 1 | Left eye inner | 18 | Right pinky (pose) |
| 2 | Left eye | 19 | Left index (pose) |
| 3 | Left eye outer | 20 | Right index (pose) |
| 4 | Right eye inner | 21 | Left thumb (pose) |
| 5 | Right eye | 22 | Right thumb (pose) |
| 6 | Right eye outer | 23 | **Left hip** |
| 7 | Left ear | 24 | **Right hip** |
| 8 | Right ear | 25 | Left knee |
| 9 | Mouth left | 26 | Right knee |
| 10 | Mouth right | 27 | Left ankle |
| 11 | **Left shoulder** | 28 | Right ankle |
| 12 | **Right shoulder** | 29 | Left heel |
| 13 | Left elbow | 30 | Right heel |
| 14 | Right elbow | 31 | Left foot index |
| 15 | Left wrist | 32 | Right foot index |
| 16 | Right wrist | | |

> Landmarks **23** (left hip) and **24** (right hip) are important — they are used as the body center reference point for **localization**.

### Hand Landmarks (21 Points per Hand)

Each hand has 21 landmarks following the standard hand skeleton:

```
        [4]                      ← Thumb tip
         |
        [3]
         |
        [2]
         |
        [1]
         |
[8] [12] [16] [20]              ← Fingertips (index, middle, ring, pinky)
 |    |    |    |
[7] [11] [15] [19]
 |    |    |    |
[6] [10] [14] [18]
 |    |    |    |
[5]--[9]--[13]-[17]             ← Knuckles (connected by palm lines)
  \   |   /   /
   \  |  /   /
    [0]                          ← Wrist
```

---

## Extraction Methods

The user can choose one of four landmark extraction methods:

### 1. None
No landmark extraction is performed. Returns an array of zeros.

### 2. MediaPipe Pose
Extracts only the **33 pose landmarks** (body skeleton). Hand landmark slots are filled with zeros.

**Use case:** When you only care about body posture and arm movement, not detailed hand shapes.

### 3. MediaPipe Hands
Extracts only the **left and right hand landmarks** (21 each). Pose landmark slots are filled with zeros.

**Use case:** When you only care about hand shapes and finger positions (e.g., fingerspelling).

### 4. MediaPipe Pose + Hands
Extracts **all 258 features** — full pose skeleton plus both hands. This is the recommended method for sign language recognition.

**Use case:** Full sign language recognition where both body movement and hand shape matter.

---

## How Extraction Works (Step by Step)

1. **Model initialization:**
   - On first use, model files are automatically downloaded from Google's servers to the `.models/` cache directory:
     - `pose_landmarker_lite.task` (~4 MB) — for pose detection
     - `hand_landmarker.task` (~6 MB) — for hand detection
   - The PoseLandmarker and/or HandLandmarker are created using MediaPipe's Tasks API.

2. **Per-frame processing:**
   - The RGB frame is wrapped in a `mediapipe.Image` object.
   - The landmarker's `.detect()` method runs the model on the image.
   - Pose results contain a list of detected people (we take the first one, `pose_landmarks[0]`).
   - Hand results contain a list of detected hands with their **handedness** labels ("Left" or "Right"), allowing correct assignment.

3. **Missing landmark handling:**
   - If no pose is detected in a frame → 132 zeros.
   - If no left hand is detected → 63 zeros (left hand slot).
   - If no right hand is detected → 63 zeros (right hand slot).
   - This zero-filling ensures every frame has exactly 258 values.

4. **Cleanup:** Landmarker objects are closed after processing to release resources.

---

## Post-Processing Options

### Normalization

**What it does:** Scales all landmark values so each feature is in the range [0, 1].

**How it works:**
```
normalized[i] = (value[i] - min[i]) / (max[i] - min[i])
```
Where `min` and `max` are computed across all frames for each feature dimension.

**Why use it:** Removes body-size variance. A tall person and a short person making the same sign will produce similar values because the coordinates are scaled relative to each person's own range.

**Important:** Normalization is applied to the **exported data** (numpy array / `.npy` file) only. The visualization always uses raw coordinates so the skeleton draws correctly on the image.

---

### Localization

**What it does:** Translates all landmarks so the **body center** is at the origin `(0, 0, 0)`.

**How it works:**
1. Find the body center = average of left hip (landmark 23) and right hip (landmark 24):
   ```
   center = (left_hip_xyz + right_hip_xyz) / 2
   ```
2. Subtract this center from the `(x, y, z)` coordinates of every landmark (pose and hands).
3. The `visibility` value of pose landmarks is not affected.

**Why use it:** Makes the data **translation-invariant**. If the signer stands on the left side of the frame vs. the right side, the landmark values will still be the same because everything is relative to the body center.

**Requirements:** Only works when pose landmarks are available (methods: "MediaPipe Pose" or "MediaPipe Pose + Hands").

**Important:** Like normalization, localization is applied to the exported data only. Localized coordinates can be negative (positions to the left of or above the body center), which would produce incorrect pixel positions for visualization.

---

## Visualization

After landmarks are extracted, the app can draw them overlaid on the original keyframe images.

**Color coding:**
- 🟢 **Green** — Pose skeleton (body joints connected by lines)
- 🟠 **Orange** — Left hand landmarks and connections
- 🔵 **Blue** — Right hand landmarks and connections

**How it works:**
1. The flat 258-value array is reshaped back into per-landmark coordinates.
2. Normalized `(x, y)` values (in [0, 1]) are multiplied by the frame's `(width, height)` to get pixel positions.
3. Connections between landmarks are drawn as lines using OpenCV.
4. Each landmark point is drawn as a filled circle with an outline.
5. Zero-valued landmarks (not detected) are skipped.

**Important:** Visualization always uses **raw** (untransformed) landmarks, not the normalized/localized version, because the raw coordinates map directly to pixel positions.

### Pose Connections

The skeleton is drawn using these anatomical connections:
- **Face:** Nose → eyes → ears, mouth corners
- **Torso:** Shoulders connected to each other and to hips
- **Arms:** Shoulder → elbow → wrist → fingers
- **Legs:** Hip → knee → ankle → heel → foot

### Hand Connections

Each hand's 21 landmarks are connected following the finger structure:
- **Thumb:** Wrist → 4 joints to tip
- **Index/Middle/Ring/Pinky:** Wrist → 4 joints each to fingertip
- **Palm:** Knuckle connections across the base of the fingers

---

## Data Export

**File:** `data_exporter.py`

### Padding / Truncation

The function `landmarks_to_numpy(landmarks, num_frames=30)` ensures the output always has exactly `num_frames` rows:

| Input frames | Action |
|---|---|
| Exactly 30 | No change |
| Fewer than 30 | Zero-padded at the end |
| More than 30 | Uniformly sampled down to 30 |

### Saving as `.npy`

The function `save_landmarks_as_npy(data, video_name)`:
1. Opens a folder picker dialog (tkinter).
2. Saves the numpy array as `{video_name}.npy` in the selected folder.
3. The saved file can be loaded with `np.load("file.npy")` and will have shape `(30, 258)`.

---

## Technical Details

### MediaPipe Tasks API

This project uses **MediaPipe 0.10.32+**, which exclusively uses the **Tasks API** (the legacy `mp.solutions` API was removed). Key differences:

- Models must be explicitly loaded from `.task` files (not bundled implicitly).
- Each image must be wrapped in `mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)`.
- Landmarkers are created via `create_from_options()` and must be explicitly `.close()`d.
- Results use lists instead of named attributes (e.g., `result.pose_landmarks[0]` instead of `result.pose_landmarks.landmark`).

### Constants

```python
NUM_POSE_LANDMARKS = 33         # Body landmarks
POSE_VALUES_PER_LANDMARK = 4    # x, y, z, visibility
POSE_FEATURE_COUNT = 132        # 33 × 4 = 132

NUM_HAND_LANDMARKS = 21         # Per hand
HAND_VALUES_PER_LANDMARK = 3    # x, y, z
HAND_FEATURE_COUNT = 63         # 21 × 3 = 63

TOTAL_FEATURES = 258            # 132 + 63 + 63
```
