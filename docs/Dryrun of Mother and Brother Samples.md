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
