"""
Preprocessing pipeline for WLASL sign language video data.

Transforms raw videos into standardized, feature-rich training data:
  - Video normalization (ffmpeg → CFR 25fps)
  - Frame extraction and temporal trimming
  - Signer localization and cropping
  - Denoising and color normalization
  - Keypoint extraction (MediaPipe Pose + Hands)
  - Optical flow computation (RAFT)
  - Motion mask generation
  - Sequence length normalization
  - Caching and quality validation
"""

from .config import PipelineConfig
from .runner import PreprocessingPipeline
