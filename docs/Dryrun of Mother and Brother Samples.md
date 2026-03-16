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
