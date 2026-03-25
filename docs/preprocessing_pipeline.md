# Preprocessing Pipeline Developer Guide

This document is the developer-facing reference for the preprocessing system under `app/preprocessing/`.

If you want the fuller narrative walkthrough, read `PREPROCESSING_PIPELINE_REPORT.md` in the project root. This file is the quicker "how the code is organized and where to edit it" version.

---

## 1. TL;DR

The preprocessing pipeline takes a raw sign-language video and produces:

```text
outputs/preprocessed/<label>/<video_stem>.mp4
outputs/preprocessed/metadata.json
```

It does that in 9 steps:

1. probe raw FPS
2. normalize to constant FPS with `ffmpeg`
3. decode frames into RGB arrays
4. apply CLAHE
5. analyze signer with MediaPipe Pose
6. crop to signer
7. resize and letterbox to square output
8. trim idle head and tail frames
9. encode final `.mp4` and write metadata

Important:

- preprocessing writes videos, not landmark tensors
- Streamlit batch mode uses `labels.csv`
- CLI batch mode walks `.mp4` files recursively

---

## 2. Main Entry Points

### Core pipeline

- `app/preprocessing/runner.py`
- `app/preprocessing/config.py`

### User-facing entry points

- `app/views/preprocessing_page.py`
- `app/preprocessing/__main__.py`

### Support modules

- `video_normalizer.py`
- `frame_extraction.py`
- `clahe.py`
- `signer_crop.py`
- `temporal_trim.py`
- `video_writer.py`
- `storage.py`
- `quality_checks.py`

---

## 3. Call Graph

For a single video, the main path is:

```text
PreprocessingPipeline.process_single_video(...)
  -> sample_exists(...)
  -> _process(...)
     -> probe_fps(...)
     -> normalize_video(...)
     -> extract_frames(...)
     -> apply_clahe(...)
     -> analyze_signer(...)
     -> crop_frames(...)
     -> resize_with_padding(...)
     -> compute_wrist_velocity(...)
     -> detect_idle_boundaries(...)
     -> write_video(...)
     -> upsert_metadata(...)
```

For batch processing:

```text
PreprocessingPipeline.run(...)
  -> dataset_dir.rglob("*.mp4")
  -> process_single_video(...) for each video
```

For reports:

```text
PreprocessingPipeline.report()
  -> generate_dataset_report(...)
     -> check_sample(...)
```

---

## 4. Step-by-Step Reference

### Step 1: Probe raw FPS

File:

- `frame_extraction.py`

Function:

- `probe_fps(video_path)`

Input:

- raw source video path

Output:

- `raw_fps: float`

Used for:

- metadata only

---

### Step 2: Normalize and decode

Files:

- `video_normalizer.py`
- `frame_extraction.py`

Functions:

- `normalize_video(...)`
- `extract_frames(...)`

Input:

- raw video path

Output:

- `frames: list[np.ndarray]` in RGB order

Notes:

- normalization is active in the current runner
- normalized output is temporary inside `_process()`
- `normalized_video_dir` is not used by the main runner

---

### Step 3: CLAHE

File:

- `clahe.py`

Function:

- `apply_clahe(frames, cfg)`

Input:

- RGB frame list

Output:

- RGB frame list with lighting normalization

Notes:

- applies CLAHE on the LAB luminance channel only
- done before crop planning

---

### Step 4: Signer analysis

File:

- `signer_crop.py`

Function:

- `analyze_signer(frames, cfg)`

Input:

- RGB frames after CLAHE

Output:

1. `bbox: tuple[int, int, int, int]`
2. `wrist_positions: np.ndarray` shaped `(T, 4)`
3. `hand_in_frame: np.ndarray` shaped `(T,)`

Notes:

- uses MediaPipe Pose in `IMAGE` mode
- estimates a forehead proxy
- estimates fingertip reach from elbow-to-wrist direction
- computes an adaptive crop with asymmetric margins
- falls back to legacy crop logic if needed

This is the most important spatial-normalization step in the pipeline.

---

### Step 5: Crop

File:

- `signer_crop.py`

Function:

- `crop_frames(frames, bbox)`

Input:

- RGB frames
- pixel bbox

Output:

- cropped RGB frames

Notes:

- intentionally simple; all crop intelligence lives in `analyze_signer()`

---

### Step 6: Resize and pad

File:

- `video_writer.py`

Function:

- `resize_with_padding(frame, target)`

Input:

- cropped RGB frame

Output:

- square RGB frame, default `512 x 512`

Notes:

- preserves aspect ratio
- adds black padding instead of stretching

---

### Step 7: Temporal trimming

File:

- `temporal_trim.py`

Functions:

- `compute_wrist_velocity(...)`
- `detect_idle_boundaries(...)`

Input:

- wrist positions
- hand visibility mask
- target FPS

Output:

- `trim_start`
- `trim_end`

Current actual behavior:

- builds `idle = no_hand OR low_velocity`
- finds the first and last confirmed active regions
- adds a 2-frame buffer on both sides
- keeps the whole clip if too few frames would remain

Developer caveat:

- `trim_min_idle_duration` exists in config
- `min_idle_frames` is computed
- the current boundary detector does not directly use it in the final decision

So the implementation is active-run based, not strict duration-gated trimming.

---

### Step 8: Encode output video

File:

- `video_writer.py`

Function:

- `write_video(frames, output_path, fps, cfg)`

Input:

- trimmed RGB frames
- output path
- output FPS

Output:

- `(success, error_message)`

Notes:

- streams raw `rgb24` frames to `ffmpeg` via stdin
- writes H.264 `.mp4`
- strips audio

---

### Step 9: Write metadata

File:

- `storage.py`

Function:

- `upsert_metadata(output_dir, label, video_name, metadata)`

Input:

- output root
- label
- original filename
- metadata dict

Output:

- updated `metadata.json`

---

## 5. Main Data Contracts

### In-memory frame format

After decode and throughout most of the pipeline:

- `list[np.ndarray]`
- RGB
- dtype `uint8`
- shape `(H, W, 3)`

### Crop analysis outputs

From `analyze_signer(...)`:

- `bbox`: `(x1, y1, x2, y2)` in pixels
- `wrist_positions`: `(T, 4)` float32 `[lx, ly, rx, ry]`
- `hand_in_frame`: `(T,)` boolean

### Final output layout

```text
outputs/preprocessed/
  metadata.json
  <label>/
    <video_stem>.mp4
```

### Metadata keys

Current metadata payload includes:

- `source_video`
- `label`
- `raw_fps`
- `output_fps`
- `original_frame_count`
- `trimmed_frame_count`
- `trim_range`
- `crop_bbox`
- `crop_size`
- `output_size`
- `pipeline_version`
- `processing_timestamp`

---

## 6. `PipelineConfig` Map

`PipelineConfig` lives in `config.py`.

### Most-used settings

- `target_fps`
- `crf_quality`
- `ffmpeg_preset`
- `clahe_clip_limit`
- `output_size`
- `crop_expansion`
- `crop_cushion_px`
- `crop_hand_tip_extension`
- `trim_vel_threshold`
- `min_active_frames`

### Crop-specific settings

- `crop_visibility_threshold`
- `crop_quantile_low`
- `crop_quantile_high`
- `crop_base_margin_side`
- `crop_base_margin_top`
- `crop_base_margin_bottom`
- `crop_motion_margin_scale`
- `crop_top_raise_scale`
- `crop_min_fraction`

### Less active settings

- `normalized_video_dir`
- `device`

Notes:

- `normalized_video_dir` is used by helper code, not by the main runner
- `device` is future-facing in preprocessing; the current main path does not branch on it

---

## 7. Streamlit vs CLI Discovery

### Streamlit page

File:

- `app/views/preprocessing_page.py`

Batch source of truth:

- `labels.csv`

### CLI runner

Files:

- `app/preprocessing/__main__.py`
- `app/preprocessing/runner.py`

Batch source of truth:

- recursive `.mp4` discovery under `dataset_dir`

This difference matters when debugging dataset setup.

---

## 8. Public API Reference

### `process_single_video(...)`

Returns:

- success with metadata
- success with `skipped=True`
- failure with `error`

### `run(...)`

Returns summary:

- `total`
- `processed`
- `skipped`
- `failed`
- `trimmed_count`

### `report()`

Returns:

- `list[SampleReport]`

---

## 9. Debugging Guide

### If normalization fails

Check:

- `ffmpeg` availability via `check_ffmpeg()`
- source video readability
- ffmpeg stderr from `write_video()` or normalization failures

### If cropping looks wrong

Check:

- `signer_crop.py`
- pose visibility threshold
- forehead proxy behavior
- hand-tip extension
- crop margin settings

### If trimming looks too aggressive or too weak

Check:

- `temporal_trim.py`
- `trim_vel_threshold`
- `min_active_frames`

Remember:

- `trim_min_idle_duration` is not currently enforced the way the comments suggest

### If batch behavior differs between UI and CLI

Check:

- whether `labels.csv` exists and is correct
- whether the CLI is just discovering files recursively

### If quality reports miss expected samples

Check:

- whether the sample was written into `metadata.json`
- the report is metadata-driven

---

## 10. Where to Edit for Common Changes

### Change encoding behavior

Edit:

- `video_normalizer.py`
- `video_writer.py`
- `PipelineConfig`

### Change crop behavior

Edit:

- `signer_crop.py`

### Change trim behavior

Edit:

- `temporal_trim.py`

### Change metadata schema

Edit:

- metadata creation in `runner.py`
- persistence in `storage.py`
- report readers in `quality_checks.py`

### Change UI controls

Edit:

- `app/views/preprocessing_page.py`

---

## 11. Known Caveats

- older project notes may still describe `25 FPS`; current default is `30`
- trimming comments are slightly ahead of the actual implementation
- preprocessing outputs videos, not model-ready tensors
- quality reports only check samples recorded in `metadata.json`

---

## 12. Summary

If you only remember one thing, remember this:

The preprocessing package is the "clean and standardize the video" stage of the project. It makes the later keyframe and landmark stages much easier by producing consistent, signer-focused clips and a dataset-wide metadata index.
