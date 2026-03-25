# Pose2Word

Word-Level Sign Language Recognition Workbench

MediaPipe Tasks • OpenCV • PyTorch • Streamlit • NumPy • UV

Pose2Word is an end-to-end project for preparing sign-language video data, extracting compact motion-aware features, training classifiers, and running upload-and-predict inference from a single workflow.

---

## Project Overview

Pose2Word is a practical pipeline for word-level sign recognition.

Instead of treating sign recognition as only a training script or only a demo UI, this repo combines:

- data preprocessing for raw videos
- semantic keyframe extraction
- landmark tensor generation
- model training and checkpointing
- interactive prediction in Streamlit

The current production path is landmark-first and centered on a 168-dimensional Relative Quantization (RQ) feature representation.

---

## The Problem We Solve

Sign-language ML projects often break in real usage because the pipeline is fragmented:

- raw videos are inconsistent (fps, framing, lighting, idle frames)
- feature generation scripts and training scripts drift from inference logic
- class order and sequence-length mismatches silently produce wrong labels
- teams juggle many disconnected tools and ad hoc scripts

Result: experiments are hard to reproduce, checkpoints are hard to trust, and inference quality drops in the real world.

---

## Our Solution

Pose2Word unifies the workflow into one coherent system.

| Stage | What it does | Output |
|---|---|---|
| Video Preprocessing | Normalize FPS, enhance contrast, crop signer, trim idle frames, resize/pad | Cleaned `.mp4` clips + `metadata.json` |
| Keyframe Extraction | Select semantically meaningful frames using hybrid motion/hand signals | `frame_*.png` folders |
| Landmark Extraction | Convert keyframes into fixed-shape landmark tensors | `.npy` tensors (default `(15, 168)`) |
| Training | Train LSTM/Transformer/Hybrid classifiers | `.pth` checkpoints + logs/history |
| Prediction | Upload video, run aligned preprocessing + feature path, classify | Top-k predictions + confidence |

---

## Why This Is Required

For sign recognition, model quality depends as much on data consistency as on architecture.

Pose2Word is built to enforce that consistency across the full lifecycle:

- same preprocessing assumptions for dataset prep and inference
- explicit feature contracts (`168` current, `258` legacy support)
- deterministic output structure for downstream training
- checkpoint-aware inference with class-order and sequence controls

This reduces hidden mismatch bugs and makes experiments easier to repeat and improve.

---

## What We Want To Do (Project Direction)

Current direction and near-term goals:

1. Improve reliability and reproducibility of the full data-to-prediction loop.
2. Expand class coverage and sample diversity beyond the current small-scale setup.
3. Tighten training/inference parity (especially sequence length and label-order ergonomics).
4. Add better evaluation reporting and comparison tooling across checkpoints.
5. Continue optimizing the landmark-first baseline before introducing heavier alternatives.

Longer-term ambition:

- evolve Pose2Word into a robust, practical sign-language experimentation platform that supports rapid iteration and cleaner deployment paths.

---

## Current Architecture

### Training-data path

```text
raw videos
	-> preprocessing pipeline
	-> preprocessed clips (.mp4)
	-> keyframe extraction
	-> keyframe image folders
	-> landmark extraction
	-> .npy tensors
	-> model training
	-> checkpoint (.pth)
```

### Inference path

```text
uploaded video
	-> in-memory preprocessing subset
	-> feature extraction
	-> checkpoint load + forward pass
	-> ranked class probabilities
```

---

## App Features (Streamlit)

Run the UI from `main.py` to access 4 tabs:

1. `Video Preprocessing`
2. `Keyframe Extractor`
3. `Landmark Extractor`
4. `Predict Sign`

What you can do in-app:

- preprocess single videos or dataset batches
- preview keyframes and motion statistics
- extract and save landmark tensors
- run closed-vocabulary prediction from a trained checkpoint

---

## Tech Stack

- Python 3.13+
- Streamlit
- MediaPipe Tasks
- OpenCV
- PyTorch / TorchVision / TorchAudio
- NumPy / SciPy / scikit-learn
- imageio-ffmpeg (bundled ffmpeg fallback)
- UV (recommended environment + dependency manager)

---

## Repository Structure

```text
app/
	core/
		inference.py
		keyframe_pipeline.py
		landmark_pipeline.py
		mediapipe_tasks.py
	preprocessing/
		runner.py
		config.py
		...
	views/
		preprocessing_page.py
		keyframe_page.py
		landmark_page.py
		predict_page.py

model/
	dataset.py
	sign_classifier.py
	trainer_current_config-gpu.py
	trainer.py

util_scripts/
	run_full_pipeline.py
	demo_pipeline.py

docs/
	pipeline_flow.md
	preprocessing_pipeline.md
	keyframe_extraction_pipeline.md
	landmark_extraction_pipeline.md
```

---

## Getting Started

### 1) Install UV (recommended)

See UV installation guide: https://docs.astral.sh/uv/getting-started/installation/

### 2) Sync dependencies

```bash
uv sync
```

### 3) Launch the app

```bash
uv run streamlit run main.py
```

---

## Typical Workflows

### A) UI-first workflow

1. Use `Video Preprocessing`
2. Use `Keyframe Extractor`
3. Use `Landmark Extractor`
4. Train from generated landmark tensors (CLI)
5. Use `Predict Sign` with the trained checkpoint

### B) End-to-end CLI workflow

```bash
uv run python util_scripts/run_full_pipeline.py \
	--dataset-dir dataset/raw_video_data \
	--preprocessed-dir outputs/preprocessed \
	--keyframes-dir outputs/keyframes \
	--landmarks-dir outputs/landmarks \
	--checkpoint-dir checkpoints/full_pipeline_run \
	--trainer-script model/trainer_current_config-gpu.py \
	--model-type lstm \
	--device cuda
```

Switch `--device` to `cpu` or `mps` when needed.

### C) Preprocessing-only CLI

```bash
uv run python -m app.preprocessing --dataset-dir dataset/raw_video_data --output-dir outputs/preprocessed
```

---

## Prediction Notes (Important)

- Prediction is closed-vocabulary: it only recognizes classes seen during training.
- Class label order must match training order exactly.
- Sequence length in prediction must match training configuration.
- If ffmpeg is unavailable, normalization/prediction preprocessing will fail.

---

## We Kept Words (Dataset Snapshot)

Based on WLASL, we kept 15 word classes:

| Word (Class) | Videos (.mp4) |
|---|---:|
| brother | 11 |
| call | 12 |
| drink | 15 |
| go | 15 |
| help | 14 |
| man | 12 |
| mother | 11 |
| no | 11 |
| short | 13 |
| tall | 13 |
| what | 12 |
| who | 14 |
| why | 11 |
| woman | 11 |
| yes | 12 |
| **TOTAL MP4s** | **187** |

Some checkpoints and local experiments in this repository may use smaller subsets.

---

## Known Constraints

- Current strongest path is landmark-first; RAFT-related artifacts exist but are not the primary active route.
- Training is CLI-driven (no dedicated training tab in the app).
- Different defaults across stages (for example sequence length) can cause accidental mismatch if not set carefully.

---

## Team

IUT-SWE-22 | CSE-4544 ML Lab Final Project

- Navid
- Musaddiq
- Mahbub
- Raiyan
- Sinha

---

## License

No license file is currently included in this repository. Add one before external distribution.
