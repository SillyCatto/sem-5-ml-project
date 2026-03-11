"""
Pipeline configuration — all tunable parameters in one place.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    """All preprocessing pipeline parameters."""

    # --- Paths ---
    dataset_dir: Path = Path("dataset/raw_video_data")
    labels_csv: Path = Path("dataset/raw_video_data/labels.csv")
    output_dir: Path = Path("preprocessed")
    normalized_video_dir: Path = Path("preprocessed/_normalized_videos")

    # --- Step 1: Video Normalization (ffmpeg) ---
    target_fps: int = 25
    crf_quality: int = 20
    ffmpeg_preset: str = "medium"

    # --- Step 3: Temporal Trimming ---
    trim_threshold_factor: float = 1.0  # motion threshold = mean + factor * std
    trim_buffer_frames: int = 2  # frames to keep before/after active segment
    min_active_frames: int = 10  # minimum frames after trimming

    # --- Step 4: Signer Localization ---
    crop_expansion: float = 1.2  # bbox expansion factor
    crop_short_side: int = 256  # short side of cropped frames (px)
    pose_sample_step: int = 5  # run pose detection every N frames

    # --- Step 5: Denoising ---
    denoise_h: int = 6  # filter strength (lower = milder)
    denoise_h_color: int = 6

    # --- Step 6: Keypoint Extraction ---
    keypoint_smooth_window: int = 5  # Savitzky-Golay window (must be odd)
    keypoint_smooth_order: int = 2  # polynomial order

    # --- Step 7: Optical Flow (RAFT) ---
    flow_model: str = "small"  # "small" or "large"
    flow_batch_size: int = 4

    # --- Step 8: Motion Masks ---
    mask_threshold_sigma: float = 1.0  # flow threshold = mean + sigma * std

    # --- Step 10: Sequence Length ---
    target_sequence_length: int = 32

    # --- Device ---
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    def resolve_device(self) -> str:
        """Determine the best available device."""
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
