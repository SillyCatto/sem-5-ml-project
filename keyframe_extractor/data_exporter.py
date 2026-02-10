"""
Data export utilities for the WLASL Keyframe & Feature Extractor.

Converts extracted landmarks to numpy arrays and saves them as .npy files.
"""

import os
import tkinter as tk
from tkinter import filedialog

import numpy as np


def landmarks_to_numpy(
    landmarks: np.ndarray,
    num_frames: int = 30,
) -> np.ndarray:
    """
    Pads or truncates a landmark array to exactly `num_frames` rows.

    If the input has fewer frames, zero-pad the remaining rows.
    If the input has more frames, uniformly sample `num_frames` from it.

    Args:
        landmarks: np.ndarray of shape (N, 258).
        num_frames: Target number of frames (default 30, matching notebook).

    Returns:
        np.ndarray of shape (num_frames, 258).
    """
    current_frames, num_features = landmarks.shape

    if current_frames == num_frames:
        return landmarks

    if current_frames < num_frames:
        # Pad with zeros
        padding = np.zeros((num_frames - current_frames, num_features), dtype=np.float32)
        return np.vstack([landmarks, padding])

    # More frames than needed: uniformly sample
    indices = np.linspace(0, current_frames - 1, num_frames, dtype=int)
    return landmarks[indices]


def save_landmarks_as_npy(
    data: np.ndarray, video_name: str
) -> tuple[bool, str]:
    """
    Opens a folder picker dialog and saves landmark data as a .npy file.

    Args:
        data: np.ndarray of shape (30, 258).
        video_name: Original video filename (used for naming the .npy file).

    Returns:
        (success: bool, message: str)
    """
    # Open Folder Picker
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    target_dir = filedialog.askdirectory(title="Select Folder to Save Landmarks (.npy)")
    root.destroy()

    if not target_dir:
        return False, "No folder selected."

    video_clean_name = os.path.splitext(video_name)[0]
    filename = f"{video_clean_name}.npy"
    full_path = os.path.join(target_dir, filename)

    try:
        np.save(full_path, data)
        return True, f"Successfully saved landmarks ({data.shape}) to: {full_path}"
    except Exception as e:
        return False, f"Error saving file: {e}"
