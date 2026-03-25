# Pose2Word Codebase Report

This report is based on the source code in the repository, starting from `main.py` and tracing runtime behavior through the app, pipelines, training code, and utility scripts. Existing markdown documentation was intentionally ignored while producing this report.

## 1. What This Project Is

Pose2Word is an end-to-end word-level sign-language tooling project built around WLASL-style videos. In its current state, the repository is best understood as a landmark-first pipeline with four user-facing jobs:

1. Standardize raw sign-language videos into trimmed, square, model-friendly `.mp4` clips.
2. Extract a small set of semantically important keyframes from those clips.
3. Convert those keyframes into fixed-shape landmark tensors for training.
4. Load a trained checkpoint and classify an uploaded sign video.

That means the repository is not only a model package and not only a Streamlit app. It is a pipeline workbench:

- `app/` contains the production-facing UI and the actual preprocessing, feature-extraction, and inference logic.
- `model/` contains dataset loaders, model definitions, and training loops.
- `util_scripts/` contains orchestration and smoke-test utilities.

Although several modules and docstrings still mention RAFT optical flow, the active pipeline that actually connects the app, landmark extraction, current training path, and inference path is centered on MediaPipe landmarks and the 168-dimensional Relative Quantization representation.

## 2. Current End-to-End Architecture

### Training-data path

```text
raw videos
  -> preprocessing pipeline
  -> preprocessed square videos
  -> keyframe extraction
  -> per-video keyframe folders
  -> landmark extraction
  -> per-video .npy tensors
  -> training
  -> checkpoint (.pth)
```

### Prediction path

```text
uploaded video
  -> preprocessing subset
  -> feature extraction
  -> checkpoint load
  -> ranked class probabilities
```

### Main package responsibilities

- `app.views`
  Streamlit pages for preprocessing, keyframes, landmarks, and prediction.
- `app.preprocessing`
  Video normalization, frame decoding, CLAHE, signer analysis, cropping, resizing, trimming, output writing, metadata, and quality checks.
- `app.core.keyframe_pipeline`
  Hybrid keyframe extraction using MediaPipe hand landmarks plus Farneback optical flow.
- `app.core.landmark_pipeline`
  MediaPipe extraction, dominant-hand normalization, shoulder calibration, Relative Quantization, and fixed-length sequence generation.
- `app.core.inference`
  Upload-to-prediction path, including checkpoint reconstruction and feature generation.
- `model`
  Dataset loading, model definitions, and training utilities.
- `util_scripts`
  Full pipeline runner, demo scripts, and environment verification helpers.

## 3. Execution Flow From `main.py`

`main.py` is the Streamlit entrypoint. It imports four page modules from `app.views`, sets page configuration and tab styling, and renders four tabs:

1. `Video Preprocessing`
2. `Keyframe Extractor`
3. `Landmark Extractor`
4. `Predict Sign`

Each page module exposes a `render()` function. Since the app runs under Streamlit, each page keeps UI state in `st.session_state` and only performs expensive work after explicit button clicks.

There is no manual keyframe selection tab in the current `main.py`. Any documentation that describes such a tab is historical rather than current behavior.

## 4. Preprocessing Page and Pipeline

### UI entrypoint

`main.py -> app.views.preprocessing_page.render()`

The preprocessing page is a UI wrapper around `app.preprocessing.PreprocessingPipeline`. It supports:

- single-video processing
- batch dataset processing
- dataset quality reporting

Single-video mode can use either:

- an uploaded file, or
- a dataset sample chosen from `labels.csv`

Batch mode in the Streamlit page reads `labels.csv` and processes the rows it finds there. This is different from the CLI batch flow, which walks `.mp4` files under the dataset directory directly.

### Core preprocessing contract

`PreprocessingPipeline.process_single_video(...)` returns one of three shapes:

- success with metadata
- success with `skipped=True` if an output already exists
- failure with an `error` string

Successful processing writes:

- `outputs/preprocessed/<label>/<video_stem>.mp4`
- or the configured equivalent under another output root

It also updates a shared:

- `outputs/preprocessed/metadata.json`

### The actual 9-step preprocessing path

`app/preprocessing/runner.py` orchestrates the current pipeline:

1. Probe raw FPS from the source video with OpenCV.
2. Normalize the video to constant frame rate H.264 with `ffmpeg`, write it to a temporary file, then decode all frames from that normalized file.
3. Apply CLAHE to the luminance channel of every frame.
4. Run MediaPipe Pose on each frame to estimate a robust signer crop and collect wrist-based trim signals.
5. Crop all frames to the chosen signer bounding box.
6. Resize and letterbox every frame to a square canvas, default `512 x 512`.
7. Trim idle head and tail segments using wrist velocity plus hand visibility.
8. Encode the result as H.264 `.mp4`.
9. Upsert shared per-sample metadata into `metadata.json`.

### Important implementation details

- Video normalization prefers a system `ffmpeg` and falls back to `imageio-ffmpeg`.
- Frames are kept in memory after decode; the pipeline does not spill intermediate images to disk.
- The crop stage is motion-aware. It uses body anchors, wrist motion, a forehead proxy, and estimated fingertip reach so raised hands are less likely to be clipped.
- Temporal trimming combines hand visibility and wrist velocity, then looks for confirmed active regions from the front and back. The code computes `trim_min_idle_duration`, but the current boundary detector does not explicitly enforce a minimum idle-run length.
- Output writing streams raw RGB frames into `ffmpeg` via stdin rather than creating temporary image files.

### Metadata and quality checks

The metadata written per sample includes at least:

- source filename
- label
- raw FPS
- output FPS
- original frame count
- trimmed frame count
- trim range
- crop bounding box
- crop size
- output size
- pipeline version
- processing timestamp

Quality reports are built from `metadata.json` plus the actual `.mp4` outputs. The report checks that each recorded output file exists, can be opened, and still has at least the minimum required frame count.

## 5. Keyframe Extraction Page and Pipeline

### UI entrypoint

`main.py -> app.views.keyframe_page.render()`

The keyframe page wraps `app.core.keyframe_pipeline`. It supports:

- single uploaded video extraction
- batch processing of a directory structured as `label/video.mp4`

The result preview shows selected frames, frame indices, timestamps, fused signal scores, and hand-detection coverage.

### What the current algorithm does

`app/core/keyframe_pipeline.py` implements a four-signal hybrid selector:

1. Farneback optical-flow magnitude inside the hand region.
2. Wrist velocity derived from MediaPipe hand landmarks.
3. Handshape change derived from finger-joint movement.
4. Optical-flow direction reversal to capture turning or circular motion.

The pipeline is:

1. Load all frames from the video.
2. Run `HandVideoLandmarker` in MediaPipe Tasks `VIDEO` mode across the clip.
3. Convert detections into left/right hand landmark vectors and combined hand bounding boxes.
4. Compute wrist-velocity and handshape-change signals.
5. Compute dense Farneback flow between consecutive frames, masked to the hand region when possible.
6. Normalize and smooth the signals.
7. Build a fused transition signal and a fused hold signal.
8. Select frames from transition peaks, hold peaks, and "sandwiched holds".
9. Force the final selection count into the configured range, default `8..15`.

### Output contract

A single run returns `PipelineResult`, which includes:

- selected keyframe indices
- keyframe times in seconds
- keyframe images
- FPS
- total frame count
- fused transition signal
- fused hold signal
- number of frames where at least one hand was detected

Batch output is written as:

```text
outputs/keyframes/
  <label>/
    <video_stem>/
      frame_0.png
      frame_1.png
      ...
```

The saved filenames are ranked order, not original frame numbers. The original frame indices are preserved in memory during the run and shown in the UI.

## 6. Landmark Extraction Page and Pipeline

### UI entrypoint

`main.py -> app.views.landmark_page.render()`

The landmark page wraps `app.core.landmark_pipeline`. It supports:

- single-directory extraction from a folder of keyframe images
- batch processing of a directory tree produced by the keyframe extractor

### What the current algorithm does

This pipeline is the clearest representation of the project's current feature format.

Input:

- a folder containing `frame_*.png` or `frame_*.jpg`

Output:

- a NumPy tensor shaped `(target_sequence_length, 168)`

The pipeline stages are:

1. Load ordered keyframe images from disk.
2. Run a composed MediaPipe Tasks detector in `IMAGE` mode:
   - pose landmarker
   - hand landmarker
   - face landmarker
3. Select exactly 56 landmarks per frame:
   - 21 left-hand landmarks
   - 21 right-hand landmarks
   - 8 pose anchors
   - 6 face anchors
4. Flatten them into a 168-dimensional vector.
5. Detect the dominant hand from activity over the whole sequence.
6. If the sequence is left-dominant, mirror it into a right-dominant canonical layout.
7. Translate all coordinates so the first-frame shoulder midpoint becomes the origin.
8. Apply Relative Quantization:
   - hands relative to their wrists
   - pose limbs relative to the shoulder midpoint
   - face anchors relative to the nose
   - shoulders kept as calibrated global anchors
9. Scale features by a constant factor, default `x100`.
10. Pad or truncate to a fixed sequence length, default `15`.

### Output layout

Batch mode writes:

```text
outputs/landmarks/
  <label>/
    <video_stem>.npy
```

The current training path is designed around these flattened per-video `.npy` tensors.

## 7. Prediction Page and Inference Path

### UI entrypoint

`main.py -> app.views.predict_page.render()`

The prediction page is the app surface for upload-and-classify inference. It asks the user for:

- a video upload
- a checkpoint path
- a confidence threshold
- a target sequence length
- an ordered class-label list

It then calls `app.core.inference.predict_video(...)`.

### What `predict_video(...)` does

1. Load the checkpoint.
2. Rebuild the matching model architecture from checkpoint config.
3. Resolve class names from manual labels, the checkpoint's `landmarks_dir`, or the built-in defaults.
4. Run the uploaded video through the preprocessing subset used for inference.
5. Extract features in one of two formats:
   - `168`-dimensional RQ landmarks, the default and current path
   - `258`-dimensional legacy pose-plus-hands keypoints, for older checkpoints
6. Run the model and compute class probabilities.
7. Return the top-k predictions plus preprocessing metadata.

### Important inference caveats

- The predictor is closed vocabulary. It can only predict the classes the checkpoint was trained on.
- The label order matters. If the manual labels do not match training order exactly, the prediction names are wrong even if the model scores are correct.
- The current UI defaults `Target sequence length` to `32`, while the current landmark extraction pipeline defaults to `15`. The value in the UI must match the length used during training.
- Inference reuses the preprocessing logic but does not write output videos or update `metadata.json`.

## 8. Training Code

### Dataset loading

`model/dataset.py` loads `.npy` feature files grouped by class. It supports two layouts:

1. current flat layout:
   `class/video.npy`
2. older nested layout:
   `class/sample/keypoints.npy`

That compatibility is useful because the repository contains traces of both the older and newer data pipeline designs.

### Models

`model/sign_classifier.py` defines three sequence classifiers:

- `LSTMSignClassifier`
- `TransformerSignClassifier`
- `HybridSignClassifier`

All three are landmark-only models in practice. The code no longer fuses RAFT features into the model forward pass, even though some module headers still describe the project that way.

### Trainers

There are two trainer files with different levels of alignment to the current pipeline.

#### `model/trainer_current_config-gpu.py`

This is the trainer that best matches the current extraction path because it:

- infers sequence length and feature dimension from the dataset
- records `landmark_dim` and `target_sequence_length` in the checkpoint config
- includes CUDA-oriented memory and throughput options

This is also the trainer used by default from `util_scripts/run_full_pipeline.py`.

#### `model/trainer.py`

This is an older trainer that still assumes older defaults in several places, including a `258`-dimensional landmark expectation. It remains useful as a reference trainer, but it is not the best default for the current `168`-dimensional RQ outputs unless adapted accordingly.

## 9. Utility Scripts

### `util_scripts/run_full_pipeline.py`

This is the main orchestration script for command-line end-to-end runs. It chains:

1. preprocessing
2. keyframe extraction
3. landmark extraction
4. training

It defaults to:

- `outputs/preprocessed`
- `outputs/keyframes`
- `outputs/landmarks`
- `model/trainer_current_config-gpu.py`

This script is the clearest CLI representation of the intended current workflow.

### `util_scripts/demo_pipeline.py`

This is a demonstration script rather than a production pipeline. It mixes:

- a standalone RAFT demo
- dataset loading checks
- model-construction smoke tests

It is helpful for orientation, but it is not part of the main app path.

### `util_scripts/verify_setup.py`

This is an environment-check script aimed at a Mac/MPS setup. It is useful for local validation but should be treated as an environment helper, not as part of the production pipeline.

## 10. Legacy and Transitional Pieces

The codebase contains a few clear signs of evolution:

- several docstrings still describe a RAFT-plus-landmarks model stack
- `model/sign_classifier.py` defaults to `landmark_dim=258`
- `model/trainer.py` still fits the older 258-dimensional framing better than the new RQ pipeline
- inference still carries a legacy 258-dimensional feature path for backward compatibility
- `docs/manual_keyframe_selection.md` refers to a tool that is not present in `main.py`

These are not random contradictions. They show that the project has moved from an older pose-plus-hands or RAFT-heavy framing toward a cleaner landmark-first pipeline built around `168`-dimensional RQ features.

## 11. Practical Conclusions

The most accurate mental model of the current repository is:

- preprocessing produces normalized videos, not training tensors
- keyframe extraction reduces temporal redundancy
- landmark extraction produces the canonical training features
- current training should be aligned to those 168-dimensional RQ tensors
- prediction depends heavily on matching training-time label order and sequence length

If someone wants to use the codebase as it exists today, the safest current path is:

1. preprocess videos
2. extract keyframes
3. extract landmarks
4. train with the shape-aware GPU trainer or the full pipeline runner
5. predict with the same class order and sequence length used in training

That is what the code most consistently supports right now.
