# Pose2Word Codebase Deep-Dive Report

## 1. Executive Summary

This repository is a multi-stage sign-language processing and recognition system centered around WLASL-style word-level clips. In practical terms, the project does four major things:

1. Preprocesses raw sign videos into normalized, trimmed, square outputs.
2. Extracts semantic keyframes from videos using a hybrid motion+landmark method.
3. Converts keyframe images into fixed-size landmark tensors using a Relative Quantization style pipeline.
4. Trains and runs sequence classifiers (LSTM/Transformer/Hybrid) on landmark sequences, with a Streamlit UI for interactive workflows.

The current production path is landmark-driven. RAFT optical-flow code exists but is not integrated in the active trainer/model forward signatures.

## 2. Top-Level Runtime Entry Flow

Primary user entrypoint is `main.py` via Streamlit.

High-level runtime path:

- `main.py` configures Streamlit and creates four tabs.
- Tabs delegate to page renderers in `app/views`:
  - Video Preprocessing
  - Keyframe Extractor
  - Landmark Extractor
  - Predict Sign
- Each page calls business logic in `app/core` and `app/preprocessing`.

This means UI is thin orchestration, while computational logic lives in modular pipeline files.

## 3. End-to-End Dataflow Across the System

### 3.1 Raw video to preprocessed output

Input: raw video files plus `labels.csv` in dataset root.

Pipeline steps implemented in `app/preprocessing/runner.py`:

1. Probe FPS from source.
2. Decode frames.
3. Apply full-frame CLAHE lighting normalization.
4. Analyze signer with MediaPipe Pose to derive:
   - adaptive crop box
   - per-frame wrist positions
   - hand-visible mask
5. Crop all frames to signer bbox.
6. Resize + pad to square (default 512x512).
7. Temporal trim using dual signal:
   - hand-visibility signal
   - wrist-velocity signal
   with idle duration gating.
8. Write output MP4 through ffmpeg stdin pipe.
9. Upsert per-sample metadata into global metadata JSON.

Output layout:

- preprocessed video: output_root/label/video_stem.mp4
- metadata registry: output_root/metadata.json

### 3.2 Preprocessed video to keyframes

Implemented in `app/core/keyframe_pipeline.py`.

For each frame sequence:

- Signal 1: Farneback flow magnitude (masked by hand bbox when available)
- Signal 2: wrist velocity from MediaPipe hand landmarks
- Signal 3: handshape delta from finger joint landmark changes
- Signal 4: flow direction reversal

Signals are normalized/smoothed/fused, and keyframe selection combines:

- transition peaks
- hold peaks
- sandwiched hold minima between motion bursts
- min/max keyframe count constraints

Output per video:

- keyframe images saved as frame_0.png, frame_1.png, ...

### 3.3 Keyframes to landmark tensors

Implemented in `app/core/landmark_pipeline.py`.

Per keyframe directory:

1. Load ordered frame images.
2. Run composed MediaPipe tasks (pose + hands + face).
3. Build compact 56-landmark feature vector (168 values per frame).
4. Detect dominant hand and mirror left-dominant samples to right-dominant canonical form.
5. Calibrate coordinates to shoulder midpoint.
6. Relative quantize local groups:
   - each hand to own wrist
   - pose limbs to shoulder midpoint
   - face to nose
7. Scale features (default x100).
8. Pad/truncate to target sequence length.

Output per sample:

- label/video_stem.npy containing shape (target_len, 168)

### 3.4 Inference path for uploaded videos

Implemented in `app/core/inference.py` and exposed by `app/views/predict_page.py`.

Flow:

- uploaded clip is normalized with ffmpeg
- full preprocessing logic is re-applied in-memory
- MediaPipe pose+hands keypoints are extracted into 258-dim vectors per frame
- sequence is padded/truncated to model target length
- checkpoint is loaded, model reconstructed from config
- top-K prediction and confidence thresholding are returned

Important note: inference feature dimension is 258, while RQ landmark extraction outputs 168. Inference assumes checkpoint config contains matching `landmark_dim` and sequence settings.

## 4. Architectural Modules

## 4.1 UI layer: app/views

### `app/views/preprocessing_page.py`

- Streamlit controls for preprocessing config and modes:
  - single video
  - full batch
  - quality report
- Uses `PipelineConfig`, `PreprocessingPipeline`, and dataset quality checks.
- Tracks preview metadata and sample frame previews in session state.

### `app/views/keyframe_page.py`

- UI for single-video and batch keyframe extraction.
- Exposes keyframe tuning controls (min/max frames, smoothing, hold threshold, MP complexity).
- Calls core keyframe pipeline and persists extracted frames.

### `app/views/landmark_page.py`

- UI for single-directory and batch landmark extraction.
- Exposes sequence length, scaling, and complexity controls.
- Displays shape, detection statistics, and allows saving `.npy` output.

### `app/views/predict_page.py`

- Upload-and-predict UI.
- Accepts checkpoint path, label order override, threshold, and sequence length.
- Uses `predict_video` and reports top predictions + preprocessing metadata.

### `app/views/_folder_browser.py`

- Cross-platform folder picker wrapper using tkinter.
- Integrates browse result into Streamlit state safely.

### `app/views/__init__.py`

- Exports page modules.

## 4.2 Core pipelines: app/core

### `app/core/mediapipe_tasks.py`

- Shared wrappers/utilities around MediaPipe Tasks API.
- Auto-downloads required `.task` models into `app/models`.
- Provides two detector abstractions:
  - video-mode hand tracker for temporal keyframe signals
  - image-mode holistic-like detector (pose+hands+face) for landmark extraction

### `app/core/keyframe_pipeline.py`

- Complete keyframe extraction algorithm.
- Includes:
  - video loading
  - flow extraction
  - hand landmark signal extraction
  - signal fusion
  - sandwiched hold detection
  - keyframe selection
  - batch processing helpers

### `app/core/landmark_pipeline.py`

- Complete RQ-style feature pipeline for keyframe folders.
- Includes index mapping, hand-dominance normalization, shoulder calibration, quantization, sequence shaping, and batch wrappers.

### `app/core/inference.py`

- Checkpoint loading + model reconstruction.
- Upload-video preprocessing and feature extraction.
- Predicts class probabilities and returns structured prediction objects.

### `app/core/__init__.py`

- Package description only.

## 4.3 Preprocessing subsystem: app/preprocessing

### `app/preprocessing/config.py`

- Defines `PipelineConfig` with all preprocessing hyperparameters:
  - ffmpeg normalization settings
  - CLAHE settings
  - crop envelope controls
  - trimming thresholds
  - output resolution and device hint

### `app/preprocessing/frame_extraction.py`

- cv2 frame decode and source FPS probing utilities.

### `app/preprocessing/clahe.py`

- Full-frame CLAHE in LAB space, preserving color channels.

### `app/preprocessing/signer_crop.py`

- MediaPipe Pose pass to compute adaptive crop and trim signals.
- Robust bbox strategy combining anchor points, movement envelope, forehead proxy, and hand-tip extension.

### `app/preprocessing/temporal_trim.py`

- Computes wrist velocity and idle boundary detection with minimum-duration gating and onset/offset buffers.

### `app/preprocessing/video_writer.py`

- Aspect-preserving resize+pad and ffmpeg-based MP4 writing via stdin stream.

### `app/preprocessing/video_normalizer.py`

- ffmpeg resolution helper and normalization function for CFR re-encode.
- `check_ffmpeg()` utility used by prediction UI.

### `app/preprocessing/storage.py`

- Global metadata JSON read/write and per-sample upsert.
- Existence checks for skip logic.

### `app/preprocessing/quality_checks.py`

- Output-file validation and dataset-level report generation.

### `app/preprocessing/runner.py`

- Main orchestration class `PreprocessingPipeline`.
- Implements single-video processing and full dataset batch run.

### `app/preprocessing/__main__.py`

- CLI interface for preprocessing and report mode.

### `app/preprocessing/__init__.py`

- Public exports: `PipelineConfig`, `PreprocessingPipeline`.

## 4.4 Model/training subsystem: model

### `model/sign_classifier.py`

- Defines three sequence classifiers:
  - `LSTMSignClassifier`
  - `TransformerSignClassifier`
  - `HybridSignClassifier`
- Factory `create_model` handles model creation and device placement.
- Current forward signature uses landmark tensor only.

### `model/dataset.py`

- Dataset loader for landmark `.npy` files by class.
- Supports nested `class/sample/keypoints.npy` and flat `class/*.npy` layouts.
- Provides train/validation split utility and DataLoader creation.

### `model/trainer.py`

- Generic training loop with metrics, checkpoints, TensorBoard logging.
- Includes AMP support and scheduler integration.
- Contains a stale call path passing `flow_dir` into `create_data_loaders` (not accepted by current dataset loader).

### `model/trainer_current_config-gpu.py`

- GPU-focused trainer variant with:
  - gradient accumulation
  - optional grad clipping
  - CUDA loader optimizations
  - configurable mixed precision
- Infers `landmark_dim` and sequence length from loaded data.

### `model/raft_flow_extractor.py`

- RAFT-based optical flow extractor and pooling utilities.
- Appears to be legacy/experimental relative to current landmark-only training path.

### `model/__init__.py`

- Re-exports model and training utilities.

### `model/train_transformer.ipynb`

- Notebook for transformer training directly on landmark arrays.
- Uses a local dataset class and 168-dim input assumption.

### `model/WLASL_recognition_using_Action_Detection.ipynb`

- Exploratory notebook for landmark auditing, visualization, augmentation, and dataset quality checks.

## 4.5 Utility scripts: util_scripts

### `util_scripts/run_full_pipeline.py`

- End-to-end CLI that chains preprocessing, keyframe extraction, landmark extraction, and training.
- Dynamically imports trainer script and calls `train_model(...)`.

### `util_scripts/demo_pipeline.py`

- Demonstration script for environment checks and sample model/flow usage.
- Contains stale assumptions (expects flow input and dictionary-style dataset samples).

### `util_scripts/verify_setup.py`

- Environment validation script focused on Mac MPS + RAFT readiness.

### `util_scripts/inspect_preprocessed.ipynb`

- Notebook for deep inspection/visualization of preprocessed artifacts and keypoint quality.

### `util_scripts/run_full_pipeline.ipynb`

- Notebook orchestrator around full pipeline script execution and model sanity checks.

## 4.6 Application packaging and configuration

### `main.py`

- Streamlit app entrypoint and tab wiring.

### `app/__init__.py`

- App package marker.

### `pyproject.toml`

- Project metadata and dependency list.
- Includes MediaPipe, Torch stack, SciPy, Streamlit, and imageio-ffmpeg.
- Declares setuptools packages for app modules.

### `uv.lock`

- Lockfile for deterministic dependency resolution.

## 4.7 Model asset files

### `app/models/hand_landmarker.task`
### `app/models/face_landmarker.task`
### `app/models/pose_landmarker_lite.task`
### `app/models/pose_landmarker_full.task`

- Binary MediaPipe task assets used by runtime detectors.
- Additional assets can be auto-downloaded by wrappers when absent.

## 5. Interface Contracts and Expected Directory Structures

## 5.1 Raw dataset expected by preprocessing UI/CLI

Expected under dataset root:

- `labels.csv` with rows carrying label/path/video_name fields.
- class folders containing source videos.

## 5.2 Keyframe batch expected structure

Input for keyframe batch mode:

- root/label/video.mp4

Output:

- keyframes_root/label/video_stem/frame_0.png ...

## 5.3 Landmark batch expected structure

Input for landmark batch mode:

- keyframes_root/label/video_stem/frame_*.png

Output:

- landmarks_root/label/video_stem.npy

## 6. Important Design Characteristics

1. Clear stage separation.
- Preprocessing, keyframe extraction, landmark extraction, and training are modularized.

2. UI and algorithm decoupling.
- Streamlit pages mostly orchestrate and display; algorithm details remain in core modules.

3. MediaPipe Tasks migration.
- Wrappers abstract model download and detector behavior to mimic old holistic contracts.

4. Robust trim/crop logic.
- Crop and temporal trim rely on signals beyond naive bounding-box or velocity-only heuristics.

5. Multiple operation modes.
- Most stages support both single-sample and batch workflows.

## 7. Current Inconsistencies and Technical Risks

1. Preprocessing runner vs normalization comments.
- `PreprocessingPipeline` comments say ffmpeg normalization in-memory, but `_process()` currently decodes source video directly via cv2 and does not call `normalize_video`.
- Inference path does call `normalize_video`.
- This can create train/inference preprocessing mismatch when source videos have variable fps/encoding behavior.

2. `model/trainer.py` loader signature mismatch.
- `train_model()` passes `flow_dir` to `create_data_loaders`, but current `create_data_loaders` does not accept `flow_dir`.
- This path likely raises `TypeError` if executed as-is.

3. `util_scripts/demo_pipeline.py` stale interfaces.
- Expects dataset samples as dict with landmarks/flow keys.
- Calls models with `(landmarks, flow)` while current models accept landmarks only.
- Script appears out of date relative to active model API.

4. `util_scripts/run_full_pipeline.py` default trainer filename mismatch risk.
- Example/default uses `trainer_current_config(gpu).py` while repository file is `trainer_current_config-gpu.py`.
- This causes trainer loading failure unless explicitly overridden.

5. Mixed feature dimensions across subsystems.
- Landmark extraction outputs 168-dim features.
- Inference feature extractor builds 258-dim pose+hands features.
- Compatibility depends on checkpoint config and data pipeline consistency.

## 8. Practical Mental Model of How the Project Works

Think of the repository as two connected but partially diverged tracks:

Track A (dataset preparation):

- raw videos -> preprocessing -> keyframes -> 168-dim quantized landmarks

Track B (serving/inference):

- uploaded video -> preprocessing-like path -> 258-dim keypoints -> classifier checkpoint

The model/trainer area contains both current landmark-only path and legacy RAFT-oriented remnants. The active production UX in Streamlit uses:

- preprocessing page
- keyframe page
- landmark page
- predict page

and depends mostly on `app/preprocessing`, `app/core/keyframe_pipeline.py`, `app/core/landmark_pipeline.py`, and `app/core/inference.py`.

## 9. File-by-File Relationship Graph (Condensed)

- `main.py` -> `app/views/*`
- `app/views/preprocessing_page.py` -> `app/preprocessing/*`
- `app/views/keyframe_page.py` -> `app/core/keyframe_pipeline.py` + folder browser helper
- `app/views/landmark_page.py` -> `app/core/landmark_pipeline.py` + folder browser helper
- `app/views/predict_page.py` -> `app/core/inference.py` + ffmpeg check
- `app/core/keyframe_pipeline.py` -> `app/core/mediapipe_tasks.py`
- `app/core/landmark_pipeline.py` -> `app/core/mediapipe_tasks.py`
- `app/core/inference.py` -> `app/preprocessing/*` + `model/sign_classifier.py`
- `app/preprocessing/runner.py` -> preprocessing stage modules + storage + quality
- `model/trainer*.py` -> `model/dataset.py` + `model/sign_classifier.py`
- `util_scripts/run_full_pipeline.py` -> preprocessing runner + core batch pipelines + trainer script dynamic import

## 10. What This Project Is, Operationally

Operationally, this is a data-centric sign-language pipeline toolkit with:

- a Streamlit front-end for repeatable dataset preparation and quick inference,
- modular algorithmic components for keyframes and landmarks,
- model training infrastructure for closed-vocabulary sign classification,
- additional exploratory notebooks/scripts for diagnostics and experimentation.

The strongest and most coherent path today is the dataset-preparation side. The model/training side works, but contains visible interface drift from an earlier RAFT-plus-landmarks design toward a landmark-only pipeline.
