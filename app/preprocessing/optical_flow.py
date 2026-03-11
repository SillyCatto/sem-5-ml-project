"""
Step 7 — Optical flow computation using RAFT.

Computes dense optical flow between consecutive cropped frames using
the pre-trained RAFT model from torchvision.  Returns both raw flow
vectors and magnitude maps.
"""

import numpy as np
import torch
from torchvision.models.optical_flow import (
    raft_small,
    raft_large,
    Raft_Small_Weights,
    Raft_Large_Weights,
)


from .config import PipelineConfig


class FlowExtractor:
    """RAFT-based dense optical flow extractor."""

    def __init__(self, cfg: PipelineConfig):
        self.device = torch.device(cfg.resolve_device())
        self.batch_size = cfg.flow_batch_size

        if cfg.flow_model == "large":
            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        else:
            self.model = raft_small(weights=Raft_Small_Weights.DEFAULT)

        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def extract(
        self,
        frames: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow for consecutive frame pairs.

        Args:
            frames: List of T RGB uint8 frames, all same (H, W).

        Returns:
            flow_vectors  — (T-1, H, W, 2) float32  (u, v displacements)
            flow_magnitude — (T-1, H, W) float32
        """
        T = len(frames)
        if T < 2:
            h, w = frames[0].shape[:2]
            return (
                np.zeros((0, h, w, 2), dtype=np.float32),
                np.zeros((0, h, w), dtype=np.float32),
            )

        h, w = frames[0].shape[:2]
        # RAFT requires H and W divisible by 8 — pad if needed
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        flow_vectors = np.zeros((T - 1, h, w, 2), dtype=np.float32)
        flow_magnitudes = np.zeros((T - 1, h, w), dtype=np.float32)

        # Pre-convert all frames to tensors once
        tensors = []
        for f in frames:
            t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0  # (3, H, W)
            if pad_h or pad_w:
                t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h))
            tensors.append(t)

        # Process in batches
        for start in range(0, T - 1, self.batch_size):
            end = min(start + self.batch_size, T - 1)
            batch_img1 = torch.stack(tensors[start:end]).to(self.device)
            batch_img2 = torch.stack(tensors[start + 1 : end + 1]).to(self.device)

            # RAFT returns a list of flow predictions (one per iteration);
            # take the last (most refined) one.
            flows = self.model(batch_img1, batch_img2)
            final_flow = flows[-1].cpu().numpy()  # (B, 2, H_pad, W_pad)

            for i, rel_idx in enumerate(range(start, end)):
                uv = final_flow[i].transpose(1, 2, 0)[
                    :h, :w
                ]  # crop padding → (H, W, 2)
                flow_vectors[rel_idx] = uv
                flow_magnitudes[rel_idx] = np.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2)

        return flow_vectors, flow_magnitudes
