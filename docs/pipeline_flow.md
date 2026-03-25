# Pipeline Flow (For Users and Developers)

This document explains how Pose2Word works end-to-end, from raw videos to training and prediction.

Read it in the way that fits you best:

- If you are a user: start at **User View**
- If you are a developer: start at **Developer View**

The project has one main training-data pipeline and one main inference pipeline.

---

## User View

### What you can do in the app

Run the app:

```bash
uv run streamlit run main.py
```

Current tabs:

1. `Video Preprocessing`
2. `Keyframe Extractor`
3. `Landmark Extractor`
4. `Predict Sign`

### Training-data pipeline (what you run to prepare data)

```text
raw videos
  -> preprocessing
  -> preprocessed square clips (.mp4)
  -> keyframe extraction (images)
  -> landmark extraction (.npy)
  -> training (checkpoint .pth)
```

What each stage produces:

- Preprocessing: a cleaned `.mp4` per sample plus a dataset `metadata.json`
- Keyframes: a folder of `frame_*.png` images per video
- Landmarks: one `.npy` file per video (default shape `(15, 168)`)
- Training: a `best_model.pth` checkpoint plus logs and training history

### If you already have a checkpoint

You can skip dataset preparation and go straight to `Predict Sign`.

You still need:

- the checkpoint file path (example: `checkpoints/my_run/best_model.pth`)
- the exact class label order used during training
- a target sequence length that matches training

If any of those do not match, the app may still run, but the predicted label names can be wrong.

### Inference pipeline (what happens when you click Predict)

```text
uploaded video
  -> preprocessing (in-memory)
  -> feature extraction
  -> checkpoint load
  -> top-k predictions
```

What you need to provide:

- a `.pth` checkpoint path
- the class label order that checkpoint was trained on
- a target sequence length that matches training

### Two recommended ways to run the pipeline

**Option A: App-driven workflow**

1. Use `Video Preprocessing` on your dataset
2. Use `Keyframe Extractor` on the preprocessed videos
3. Use `Landmark Extractor` on the keyframes
4. Train using the generated landmark files (see next section)
5. Use `Predict Sign` with the trained checkpoint

**Option B: One-command CLI workflow**

```bash
uv run python util_scripts/run_full_pipeline.py \
  --dataset-dir dataset/raw_video_data \
  --preprocessed-dir outputs/preprocessed \
  --keyframes-dir outputs/keyframes \
  --landmarks-dir outputs/landmarks \
  --checkpoint-dir checkpoints/lstm_run1 \
  --trainer-script model/trainer_current_config-gpu.py \
  --model-type lstm \
  --num-classes 15 \
  --device cuda
```

Switch `--device` to `cpu` or `mps` if needed.

### Training: How to get a `.pth` checkpoint

The app does not include a training tab. Training is done via CLI scripts after you have landmark `.npy` files.

If you already have landmark files in `outputs/landmarks` (or another folder), you can run training only:

```bash
uv run python util_scripts/run_full_pipeline.py \
  --skip-preprocessing --skip-keyframes --skip-landmarks \
  --landmarks-dir outputs/landmarks \
  --checkpoint-dir checkpoints/my_run \
  --trainer-script model/trainer_current_config-gpu.py \
  --model-type lstm \
  --device cpu
```

This produces:

- `checkpoints/my_run/best_model.pth`
- `checkpoints/my_run/logs/` (TensorBoard)
- `checkpoints/my_run/training_history.json`

If you want the full end-to-end run (preprocess -> keyframes -> landmarks -> training), use Option B above without the `--skip-*` flags.

### Common user gotchas

- If you use `Batch` or `Select from dataset` in preprocessing, you need `labels.csv` under the dataset root
- The prediction tab defaults sequence length to `32`, while the landmark extractor defaults to `15`; those must match training
- Class order matters: wrong label order means correct scores but wrong label names
- The paths shown here are defaults; the UI lets you change output folders

---

## Developer View

### The two pipeline graphs

**Training data path**

```text
raw videos
  -> app.preprocessing (writes .mp4 + metadata)
  -> app.core.keyframe_pipeline (writes keyframe images)
  -> app.core.landmark_pipeline (writes .npy tensors)
  -> model/* training (writes .pth checkpoint)
```

**Inference path**

```text
uploaded video
  -> app.core.inference (runs preprocessing subset in memory)
  -> feature extraction (168-dim default, 258-dim legacy)
  -> model forward pass
  -> ranked probabilities
```

### Where each stage lives

Preprocessing:

- UI: `app/views/preprocessing_page.py`
- Pipeline: `app/preprocessing/runner.py`
- Config: `app/preprocessing/config.py`
- CLI: `app/preprocessing/__main__.py`

Keyframes:

- UI: `app/views/keyframe_page.py`
- Pipeline: `app/core/keyframe_pipeline.py`

Landmarks:

- UI: `app/views/landmark_page.py`
- Pipeline: `app/core/landmark_pipeline.py`

Inference:

- UI: `app/views/predict_page.py`
- Logic: `app/core/inference.py`

Training:

- Data: `model/dataset.py`
- Models: `model/sign_classifier.py`
- Recommended trainer: `model/trainer_current_config-gpu.py`
- Legacy-ish trainer: `model/trainer.py`

### Key defaults and knobs

Defaults you will see in the current code:

- Preprocessing output size is square `512 x 512` unless changed in `PipelineConfig`
- Keyframe selection defaults to `8..15` unless changed in `keyframe_pipeline.PipelineConfig`
- Landmark tensors default to `(15, 168)` unless changed in `LandmarkConfig`
- Inference UI defaults sequence length to `32` unless changed in the predictor tab

Where to change behavior:

- preprocessing: `app/preprocessing/config.py`, `app/preprocessing/runner.py`
- keyframes: `app/core/keyframe_pipeline.py`
- landmarks: `app/core/landmark_pipeline.py`
- inference features: `app/core/inference.py`

### Data contracts (what each stage expects)

Raw dataset for the preprocessing page (batch and dataset picker):

- dataset root plus `labels.csv`
- typical layout:

```text
dataset/raw_video_data/
  labels.csv
  <label>/
    <video>.mp4
```

Expected CSV columns:

```csv
label,video_name,path
brother,brother_1.mp4,brother/brother_1.mp4
```

Preprocessed videos:

```text
outputs/preprocessed/
  metadata.json
  <label>/
    <video_stem>.mp4
```

Keyframes:

```text
outputs/keyframes/
  <label>/
    <video_stem>/
      frame_0.png
      frame_1.png
      ...
```

Landmarks:

```text
outputs/landmarks/
  <label>/
    <video_stem>.npy
```

Checkpoints:

```text
checkpoints/<run_name>/
  best_model.pth
  logs/
  training_history.json
```

### Feature formats (why you see 168 and sometimes 258)

- Current default: `168`-dimensional RQ landmarks used by `app/core/landmark_pipeline.py`
- Legacy compatibility: `258`-dimensional pose-plus-hands keypoints supported by `app/core/inference.py` for older checkpoints

### Important developer caveats

- The repo is landmark-first; RAFT exists but is not wired into the main training/inference path
- Streamlit preprocessing batch uses `labels.csv`, but the CLI preprocessing path can walk `.mp4` files without it
- The predictor’s sequence length must match training; the UI default may not match your dataset defaults

---

## How Things Connect (One Paragraph)

Preprocessing standardizes raw videos into square, trimmed `.mp4` clips so later stages are more stable. Keyframe extraction then selects 8 to 15 informative frames per clip and writes them as `frame_*.png`. Landmark extraction turns those images into fixed-length `(T, 168)` tensors using MediaPipe Tasks plus Relative Quantization. Training consumes those `.npy` tensors to produce a checkpoint. Prediction loads that checkpoint, runs the preprocessing subset in memory on a new upload, extracts matching features, and returns ranked class probabilities.
