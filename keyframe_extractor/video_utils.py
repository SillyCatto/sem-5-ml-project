"""
Video I/O utilities for the WLASL Keyframe & Feature Extractor.
"""

import cv2


def get_frames_from_video(video_path: str) -> list:
    """Reads all frames from a video file and returns them as a list of RGB images."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames
