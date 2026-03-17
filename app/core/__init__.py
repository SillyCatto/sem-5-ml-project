"""
Core business-logic package for the WLASL Dataset Preparation Tool.

Modules
-------
keyframe_pipeline   Keyframe extraction (Farneback + MediaPipe Holistic hybrid).
landmark_pipeline   Landmark extraction (MediaPipe Holistic + RQ pipeline).
video_utils         Video I/O and the full preprocessing pipeline.
algorithms          (legacy) Old keyframe extraction algorithms.
landmark_extractor  (legacy) Old landmark extraction and visualisation.
data_exporter       NumPy array export utilities.
file_utils          Frame image saving utilities.
"""
