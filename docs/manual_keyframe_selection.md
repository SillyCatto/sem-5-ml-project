# Manual Keyframe Selection Tool

## Purpose

The manual keyframe selection tool allows human annotators to view every frame of a sign language video and hand-pick the frames that best represent meaningful sign transitions — **keyframes**. These manually selected keyframes serve as **ground truth** for evaluating the accuracy of automatic keyframe extraction algorithms (optical flow, skeleton-based, etc.).

Without ground truth, there is no way to measure whether an automatic algorithm is actually selecting the *right* frames, so this tool is a prerequisite for any quantitative evaluation.

---

## Why Not Just Use OpenCV to Read Frames?

The original implementation used OpenCV's `cv2.VideoCapture` with FPS resampling:

```python
# OLD approach — unreliable frame counts
frames = get_frames_from_video(video_path, target_fps=30.0)
```

**Problem:** OpenCV reads frames based on container metadata, which is unreliable for web-scraped videos (many are VFR — variable frame rate). For the same video `who_63229.mp4`:
- OpenCV at 30fps → **47 frames**
- Preprocessing pipeline (ffmpeg at 25fps CFR) → **49 frames**

If ground truth frame indices are based on a different frame extraction than what the preprocessor uses, the indices are meaningless — frame #20 in OpenCV might be a completely different moment than frame #20 in the preprocessor output.

**Solution:** Use the exact same tool (ffmpeg) and parameters (25fps CFR) as the preprocessing pipeline.

---

## Frame Extraction Approach

### How ffmpeg Extracts Frames

```
ffmpeg -y -i <video> -r 25 -f image2 -pix_fmt rgb24 <tmpdir>/frame_%06d.png
```

| Flag | Purpose |
|------|---------|
| `-r 25` | Re-encode to exactly 25 fps constant frame rate |
| `-f image2` | Output each frame as a separate image file |
| `-pix_fmt rgb24` | Force RGB pixel format (no alpha, no YUV) |
| `frame_%06d.png` | Sequential numbering: `frame_000001.png`, `frame_000002.png`, … |

This is functionally identical to what the preprocessing pipeline does in Step 1 (video normalization) + Step 2 (frame extraction):
1. **Preprocessor Step 1:** `ffmpeg → CFR H.264 video at 25fps`
2. **Preprocessor Step 2:** `cv2.VideoCapture → read every frame`

The manual tool combines both into one step — ffmpeg directly outputs individual frames. The frame count will match because both use ffmpeg's `-r 25` resampling on the same source video.

### Why 25 FPS?

- Matches `PipelineConfig.target_fps = 25`
- 25fps is standard for PAL video and provides a good balance between temporal resolution and data volume
- All downstream processing (optical flow, keypoint extraction, sequence normalization) expects this frame rate

### Why PNG (not JPEG) for Intermediate Frames?

The ffmpeg output uses PNG (lossless) for the temporary frame files read into memory. This avoids JPEG compression artifacts that could affect frame comparison. The final saved ground truth frames *are* saved as JPEG since they're just for human reference, not model input.

---

## Video Source Selection

### Dataset Picker (Primary)

The tool reads `dataset/raw_video_data/labels.csv` to present a dropdown of all dataset videos organized by class:

```
who/who_63229.mp4
who/who_63230.mp4
call/call_91832.mp4
...
```

This ensures annotators are selecting from the **exact same files** the preprocessor will process — no filename mismatches, no path confusion.

### Upload (Secondary)

File upload is preserved as a fallback for ad-hoc videos not yet in the dataset. Uploaded videos go through the same ffmpeg extraction pipeline.

---

## Ground Truth Storage

### Directory Structure

```
dataset/ground_truth/
├── who/
│   ├── who_63229/
│   │   ├── frames/
│   │   │   ├── frame_00003.jpg
│   │   │   ├── frame_00012.jpg
│   │   │   └── frame_00028.jpg
│   │   ├── ground_truth.json
│   │   └── ground_truth.npy
│   └── who_63230/
│       └── ...
├── call/
│   └── ...
└── ...
```

Organized as `<label>/<video_stem>/` to mirror the dataset structure and make batch comparison with automatic algorithms straightforward.

### Output Files

#### `ground_truth.json`
```json
{
  "video": "who_63229.mp4",
  "label": "who",
  "target_fps": 25,
  "native_fps": 24.0,
  "native_resolution": "1920x1080",
  "total_frames": 49,
  "selected_count": 8,
  "selected_indices": [3, 8, 12, 19, 25, 31, 38, 44]
}
```

- `target_fps` — the FPS used for extraction (for reproducibility)
- `native_fps` / `native_resolution` — original video properties (for reference)
- `total_frames` — total frames at target FPS (must match preprocessor)
- `selected_indices` — the ground truth keyframe indices (0-indexed)

#### `ground_truth.npy`

Same indices as a numpy `int32` array for direct use in evaluation scripts:

```python
gt = np.load("dataset/ground_truth/who/who_63229/ground_truth.npy")
# array([ 3,  8, 12, 19, 25, 31, 38, 44], dtype=int32)
```

#### `frames/`

JPEG images of the selected keyframes for visual inspection. Named `frame_XXXXX.jpg` where `XXXXX` is the 0-indexed frame number.

---

## How to Use the Tool

1. **Launch:** `uv run streamlit run main.py` → go to the "Manual Keyframe Selection" tab
2. **Select a video** from the dataset dropdown (or upload one)
3. **Wait** for ffmpeg to extract all frames at 25fps
4. **Review** the frame grid — every frame is shown as a thumbnail
5. **Click** individual frames to toggle selection (green border = selected)
6. **Use helpers** if needed: "Select All", "Clear All", "Every N frames"
7. **Save** — ground truth is written to `dataset/ground_truth/<label>/<video_stem>/`

If ground truth already exists for a video, the tool detects it and offers to reload the previous selection.

---

## How Ground Truth Will Be Used

The saved indices can be compared against automatic keyframe extraction results:

```python
# Example evaluation (pseudocode)
gt_indices = np.load("dataset/ground_truth/who/who_63229/ground_truth.npy")
auto_indices = run_optical_flow_extraction(video, target_count=len(gt_indices))

# Metrics
precision = len(set(auto_indices) & set(gt_indices)) / len(auto_indices)
recall = len(set(auto_indices) & set(gt_indices)) / len(gt_indices)
f1 = 2 * precision * recall / (precision + recall)

# Or use a proximity metric (how close is each auto frame to the nearest GT frame)
distances = [min(abs(a - g) for g in gt_indices) for a in auto_indices]
mean_distance = np.mean(distances)
```

This enables objective comparison between different algorithms (Farneback optical flow, Lucas-Kanade, RAFT, skeleton-based) on the same videos with the same frame extraction.
