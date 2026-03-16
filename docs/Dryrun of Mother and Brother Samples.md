# Dry Run of Mother and Brother Samples

This document captures the end-to-end dry run we completed with a small two-class sample set (brother and mother).

## Scope

- Goal: verify that preprocessing, training, checkpointing, and prediction all work end-to-end.
- Classes: brother, mother.
- Samples: 3 videos per class (6 total).
- Model: LSTM classifier.

## Data Preparation

### Folder layout used

```text
dataset/
	raw_video_data/
		labels.csv
		brother/
			brother_1.mp4
			brother_2.mp4
			brother_3.mp4
		mother/
			mother_1.mp4
			mother_2.mp4
			mother_3.mp4
```

### Labels file

- File: `dataset/raw_video_data/labels.csv`
- Format used: `label,video_name,path`
- Example rows:

```csv
label,video_name,path
brother,brother_1.mp4,brother/brother_1.mp4
mother,mother_1.mp4,mother/mother_1.mp4
```

## Preprocessing Dry Run

### What we ran

- Ran the preprocessing pipeline on the 6 videos and generated training-ready keypoints.
- Output directory used: `preprocessed_mini`

### Output structure observed

```text
preprocessed_mini/
	brother/
		brother_1/
			keypoints.npy
		brother_2/
			keypoints.npy
		brother_3/
			keypoints.npy
	mother/
		mother_1/
			keypoints.npy
		mother_2/
			keypoints.npy
		mother_3/
			keypoints.npy
```

### Important compatibility note

- The project originally expected a flat layout (`class/*.npy`).
- We updated the dataset loader to also support nested layout (`class/sample/keypoints.npy`), which matches the current preprocessing output.

## Training Dry Run

### Quick training command (5 epochs)

```powershell
uv run python -c "from model.trainer import train_model; train_model(landmarks_dir='preprocessed_mini', flow_dir=None, model_type='lstm', num_classes=2, batch_size=2, num_epochs=5, learning_rate=0.001, save_dir='checkpoints/mini_brother_mother_quick5', device='cpu')"
```

### Result summary

- Training completed in about 1.41 minutes.
- Best validation accuracy reached 1.0000 at epoch 4 (on a very small validation split).
- Best checkpoint saved at:
	- `checkpoints/mini_brother_mother_quick5/best_model.pth`

### Why this was a dry run

- Dataset size is tiny (6 total samples), so validation metrics can fluctuate heavily.
- The objective was pipeline verification, not final model quality.

## Monitoring and Runtime Checks Used

### Check if training is still running

```powershell
Get-Process python,uv -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,CPU,StartTime
```

### Check checkpoint update time and size

```powershell
Get-Item .\checkpoints\mini_brother_mother_quick5\best_model.pth | Select-Object LastWriteTime,Length
```

### How we confirmed completion

- Training terminal returned to prompt.
- No training-related Python process remained active.
- Checkpoint timestamp stopped changing.

## Prediction Dry Run (Streamlit)

### UI path used

- Opened the new `Predict Sign` tab.
- Set checkpoint path to:
	- `checkpoints/mini_brother_mother_quick5/best_model.pth`
- Set class labels to:
	- `brother,mother`

### Initial behavior observed

- The app correctly loaded the model and produced ranked predictions.
- Typical early output was near-tie confidence (example: mother 0.5013 vs brother 0.4987).
- Because confidence was near 0.50, status often appeared as `Not confident` when threshold was higher.

### Interpretation

- This is expected for a very small dataset and short training.
- End-to-end inference flow is functioning correctly.

## GitHub Dry-Run Checklist

### Files and folders included in this update

- Prediction backend and UI wiring:
	- `app/core/inference.py`
	- `app/views/predict_page.py`
	- `app/views/__init__.py`
	- `main.py`
- Training and dataset compatibility fixes:
	- `model/dataset.py`
	- `model/trainer.py`
	- `app/preprocessing/video_normalizer.py`
	- `app/preprocessing/__init__.py`
- Documentation:
	- `docs/Dryrun of Mother and Brother Samples.md`

### Items intentionally excluded from GitHub

- Runtime artifacts like checkpoints (`checkpoints/`), raw dataset videos (`dataset/`), and generated preprocessed outputs.
- These are already ignored by project gitignore rules and should stay local.

## Recommended Next Steps

1. Train a slightly longer run (20 to 50 epochs) for more stable confidence separation.
2. Re-test all 6 videos and record per-video predictions in this same document.
3. Add more samples per class before treating metrics as meaningful.
