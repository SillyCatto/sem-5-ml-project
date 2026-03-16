# Pose2Word — Complete Project Guide

**IUT-SWE-22 | CSE-4544 ML Lab Final Project**

This document explains the entire project from scratch, what has already been built, what still needs to be done, and what to do right now with your video dataset.

---

## What Are We Trying To Do?

We want to build a tool where someone can upload a short video of themselves doing an American Sign Language (ASL) sign, and the tool will tell them what word that sign means.

For example:
- User records themselves signing **"mother"**
- They upload the video
- The app says → **"mother"** (with a confidence score)

This is a **15-word closed-vocabulary classifier**. That means it only knows 15 specific ASL words right now. It will not understand any sign outside those 15 words. If it sees something unfamiliar, it should say it is not confident instead of guessing wrong.

The 15 supported words are:
> brother, call, drink, go, help, man, mother, no, short, tall, what, who, why, woman, yes

---

## How Does It Work? (From Scratch)

Think of it like a 4-step journey from a raw video to a predicted word.

### Step 1 — Get the raw video
The input is a short MP4 video of someone doing an ASL sign. We have 187 such videos (11 to 15 per word) from the WLASL public dataset.

### Step 2 — Clean and prepare the video
A raw video is noisy. The camera angle varies, the background is different, the signer is at different distances, the FPS might vary, and there are idle frames at the beginning and end. We fix all of this through a preprocessing pipeline.

The pipeline does these things in order:
1. **Normalize** the video to a constant 25 FPS using ffmpeg (consistent timing).
2. **Extract frames** as individual images.
3. **Trim idle segments** — cut out frames where the signer is not moving yet.
4. **Crop the signer** — detect the body and zoom into just the signer, ignoring the background.
5. **Denoise** — reduce visual noise in each frame.
6. **Extract keypoints** — using MediaPipe, find the positions of the body, left hand, and right hand joints in every frame. This gives a 258-number vector per frame (33 pose joints × 4 values + 21 left hand joints × 3 values + 21 right hand joints × 3 values).
7. **Optical flow** (optional) — using RAFT, compute how fast and in which direction pixels are moving between frames. This captures the motion of the sign.
8. **Normalize sequence length** — pad or subsample so every video ends up as exactly 32 frames of keypoints. The model always sees the same shape of input.

After this step, each video becomes a **32×258 numpy array** — 32 time steps, 258 keypoint features per step.

### Step 3 — Train a model
We feed these 32×258 arrays into a deep learning model. The model learns to associate the pattern of keypoint movements with a specific word.

We have three model architectures available:
- **LSTM** — fast, sequential, good for small datasets
- **Transformer** — more powerful, needs more data
- **Hybrid** — balances both

The model is trained until it can correctly predict which of the 15 words a video belongs to.

### Step 4 — Predict on new videos
When a user uploads a new video through the app, it goes through the same preprocessing steps (normalize, trim, crop, denoise, keypoints, sequence normalize), becomes a 32×258 array, and gets fed into the trained model. The model outputs 15 confidence scores, one per word. The word with the highest score is the predicted sign.

If the highest confidence score is below a threshold (default 55%), the app says it is not confident rather than returning a bad guess.

---

## What Has Already Been Built

### The Streamlit App — `main.py`
The main interface. Launch with:
```
uv run streamlit run main.py
```
It now has 5 tabs:

| Tab | Purpose |
|-----|---------|
| 🎯 Predict Sign | Upload a video, point to a checkpoint, get a prediction |
| 🔧 Video Preprocessing | Run the full pipeline on dataset videos |
| 🖱️ Manual Keyframe Selection | Pick keyframes by hand |
| 📹 Keyframe Extractor | Auto-extract keyframes from a video |
| 🦴 Landmark Extractor | Extract and save MediaPipe keypoints |

### The Preprocessing Pipeline — `app/preprocessing/`
Handles every step from raw video to clean keypoints. Each step is a separate module:

| File | What it does |
|------|-------------|
| `config.py` | All tunable settings in one place (FPS, crop size, sequence length, etc.) |
| `video_normalizer.py` | Re-encode video to constant FPS with ffmpeg |
| `frame_extraction.py` | Read frames from video, trim idle ones |
| `signer_crop.py` | Detect signer body, crop and resize |
| `denoise.py` | Reduce visual noise per frame |
| `keypoint_extraction.py` | MediaPipe Pose + Hands → 258 features per frame |
| `optical_flow.py` | RAFT optical flow computation |
| `motion_mask.py` | Generate motion-based masks |
| `sequence_normalizer.py` | Pad/subsample to fixed length |
| `storage.py` | Save results as `.npy` files on disk |
| `quality_checks.py` | Validate saved output |
| `runner.py` | Orchestrate all steps end to end |

### The Model Layer — `model/`
Handles training and classification.

| File | What it does |
|------|-------------|
| `sign_classifier.py` | 3 model architectures: LSTM, Transformer, Hybrid |
| `dataset.py` | PyTorch Dataset that loads `.npy` keypoint files |
| `trainer.py` | Training loop with validation, early stopping, checkpointing |
| `raft_flow_extractor.py` | RAFT optical flow as a standalone extractor |

### The Inference Helper — `app/core/inference.py`
The new prediction engine. It:
- Takes an uploaded video and a checkpoint path
- Runs the same preprocessing pipeline used during training
- Loads the model from the checkpoint
- Returns top-K predicted words with confidence scores
- Returns "not confident" if no prediction clears the threshold

### The Prediction Tab — `app/views/predict_page.py`
The new Streamlit UI for the end-user prediction flow:
- Upload a video
- Paste the checkpoint path
- Adjust the confidence threshold slider
- Confirm the class labels match training
- Click Predict — see the result

---

## What Is NOT Done Yet

These are the things required before the tool actually works for real users.

### 1. The model has not been trained yet — there is no `.pth` checkpoint
**This is the most critical missing piece.** The prediction tab exists but cannot produce a real prediction because there is no trained model file. Without `best_model.pth` the app will throw an error when you click Predict.

### 2. The dataset has not been preprocessed yet
The 187 raw `.mp4` videos need to be run through the preprocessing pipeline to produce `.npy` keypoint files. The model is trained on those keypoint files, not on the raw videos directly.

### 3. The train/validation split has not been enforced by signer
Right now the split is random. You should split by signer identity so no person appears in both training and test data. This prevents the model from cheating by recognizing specific people rather than the sign.

### 4. The model has not been evaluated with per-class accuracy
After training, you need to see how well each word individually performs, not just an overall number. Some words (like "man" and "woman") might be confused with each other.

### 5. No real-user testing has been done
Everything so far uses videos from the dataset. Fresh recordings from team members who appear nowhere in the dataset are needed to get an honest measure of real-world performance.

---

## What To Do Right Now With Your Existing Videos

You have the raw `.mp4` videos in a folder somewhere. Here is exactly what to do with them — in order.

### Step A — Set up the folder structure

The preprocessing pipeline expects your videos organized like this:
```
dataset/
  raw_video_data/
    labels.csv
    mother/
      mother_1.mp4
      mother_2.mp4
      ...
    brother/
      brother_1.mp4
      ...
    call/
      ...
```

If your folder is in a different location, that is fine. You will point the app at it.

`labels.csv` needs to have this format — one row per video:
```csv
label,video_name,path
mother,mother_1.mp4,mother/mother_1.mp4
brother,brother_1.mp4,brother/brother_1.mp4
```

### Step B — Run the preprocessing pipeline through the app

1. Start the app: `uv run streamlit run main.py`
2. Go to the **🔧 Video Preprocessing** tab
3. Under Pipeline Configuration, set the dataset directory to wherever your videos folder is
4. Set the output directory to `preprocessed/` (or any folder you want)
5. Switch to **📦 Batch (Full Dataset)** mode
6. Check "Skip optical flow" first to go faster (you can add flow later)
7. Click **🚀 Start Batch Processing**
8. Wait — it will process all 187 videos and save keypoints as `.npy` files under `preprocessed/`

After this you will have:
```
preprocessed/
  mother/
    mother_1/
      keypoints.npy  ← 32×258 array, the actual training data
      frames.npy
      metadata.json
  brother/
    ...
```

### Step C — Train the model

Once preprocessing is done, run training using Python directly (or we can wire a training tab into the UI later):

```python
from model.trainer import train_model

train_model(
    landmarks_dir="preprocessed",   # where your .npy files are
    flow_dir=None,                   # leave None for now
    model_type="lstm",               # start with lstm, simplest
    num_classes=15,
    batch_size=8,                    # small batch because dataset is small
    num_epochs=100,
    learning_rate=0.001,
    save_dir="checkpoints/lstm_v1"
)
```

The best model checkpoint will be saved at `checkpoints/lstm_v1/best_model.pth`.

### Step D — Predict

1. Go to the **🎯 Predict Sign** tab
2. Upload any `.mp4` video of an ASL sign
3. Set the checkpoint path to `checkpoints/lstm_v1/best_model.pth`
4. Confirm the class labels match (default 15 are pre-filled)
5. Click **Predict Sign**
6. See the result

---

## Important Things To Remember

**The label order must match training exactly.**
The model outputs index 0, 1, 2, ... and those are mapped to class names alphabetically from the `preprocessed/` folder. The default order in the app is alphabetical:
```
brother, call, drink, go, help, man, mother, no, short, tall, what, who, why, woman, yes
```
Do not change this order in the Predict tab unless your training used a different order.

**The preprocessing at prediction time must match training.**
Sequence length, FPS, crop settings — they all need to be the same. The default settings in `PipelineConfig` are what training uses. Do not change them between training and prediction.

**Low confidence means the sign was not recognized, not that the model is broken.**
If the user signs something outside the 15 supported words, the model will still pick one of the 15 — but with a low score. That is why the confidence threshold exists. Treat any result below 55% as "unrecognized".

**Accuracy on the training set is not a useful number.**
The real measure is how well the model does on videos it has never seen before — especially from signers who were not in the training set at all.

---

## Project File Map

```
Pose2Word/
├── main.py                          ← Launch the Streamlit app
├── pyproject.toml                   ← Python dependencies (uv)
│
├── app/
│   ├── core/
│   │   ├── inference.py             ← NEW: video → prediction engine
│   │   ├── landmark_extractor.py   ← MediaPipe keypoints (simple path)
│   │   ├── video_utils.py          ← Frame reading, rescaling, contrast
│   │   ├── data_exporter.py        ← Save landmarks as .npy
│   │   ├── algorithms.py           ← Keyframe detection algorithms
│   │   └── file_utils.py           ← File helpers
│   │
│   ├── preprocessing/               ← Full pipeline (dataset prep)
│   │   ├── config.py
│   │   ├── video_normalizer.py
│   │   ├── frame_extraction.py
│   │   ├── signer_crop.py
│   │   ├── denoise.py
│   │   ├── keypoint_extraction.py
│   │   ├── optical_flow.py
│   │   ├── motion_mask.py
│   │   ├── sequence_normalizer.py
│   │   ├── storage.py
│   │   ├── quality_checks.py
│   │   └── runner.py
│   │
│   └── views/
│       ├── predict_page.py          ← NEW: upload & predict UI
│       ├── preprocessing_page.py
│       ├── keyframe_page.py
│       ├── landmark_page.py
│       └── manual_keyframe_page.py
│
├── model/
│   ├── sign_classifier.py           ← LSTM / Transformer / Hybrid models
│   ├── dataset.py                   ← PyTorch dataset loader
│   ├── trainer.py                   ← Training loop + checkpointing
│   └── raft_flow_extractor.py       ← RAFT optical flow
│
└── docs/
    ├── classify_words.md            ← This file
    ├── preprocessing_pipeline.md
    ├── RAFT_GUIDE.md
    └── ...
```

---

## Summary: The Remaining Work In Order

| # | What | Why |
|---|------|-----|
| 1 | Place videos in `dataset/raw_video_data/` with `labels.csv` | Pipeline needs this structure |
| 2 | Run batch preprocessing to produce `preprocessed/*.npy` files | Model trains on keypoints, not raw video |
| 3 | Call `train_model()` to produce `best_model.pth` | No checkpoint = no predictions |
| 4 | Evaluate per-class accuracy + confusion matrix | Understand where it fails |
| 5 | Record fresh videos from team members | Test on truly unseen signers |
| 6 | Load checkpoint in Predict tab and test | End-user demo |
| 7 | Optionally: retrain with optical flow enabled | May improve accuracy |
