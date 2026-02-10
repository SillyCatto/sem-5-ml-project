"""
File saving utilities for the WLASL Keyframe & Feature Extractor.
"""

import os

import cv2


def save_frames_to_folder(
    frames: list, indices, video_name: str, target_dir: str | None = None
) -> tuple[bool, str]:
    """
    Saves extracted frames as JPEG images to the provided folder.

    Returns:
        (success: bool, message: str)
    """
    if not target_dir:
        return False, "Please provide an output folder path."

    target_dir = os.path.expanduser(target_dir)
    if not os.path.isdir(target_dir):
        return False, "Output folder does not exist."

    # Create Sub-folder
    video_clean_name = os.path.splitext(video_name)[0]
    save_path = os.path.join(target_dir, video_clean_name)

    try:
        os.makedirs(save_path, exist_ok=True)
    except OSError as e:
        return False, f"Error creating directory: {e}"

    # 3. Save Images
    count = 0
    for frame, idx in zip(frames, indices):
        # Convert RGB back to BGR for OpenCV saving
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = f"{video_clean_name}-frame-{idx}.jpg"
        full_path = os.path.join(save_path, filename)
        cv2.imwrite(full_path, bgr_frame)
        count += 1

    return True, f"Successfully saved {count} frames to: {save_path}"
