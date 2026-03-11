"""
Step 1 — Video normalization using ffmpeg.

Re-encodes raw videos to constant frame rate (CFR) H.264 for consistent
downstream processing.  VFR → CFR conversion is critical because optical
flow assumes constant inter-frame timing.
"""

import subprocess
import shutil
from pathlib import Path

import imageio_ffmpeg

from .config import PipelineConfig


def _get_ffmpeg() -> str:
    """Return path to ffmpeg — prefer system install, fall back to bundled binary."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available (system or bundled)."""
    try:
        exe = _get_ffmpeg()
        return exe is not None and len(exe) > 0
    except Exception:
        return False


def normalize_video(
    input_path: Path,
    output_path: Path,
    cfg: PipelineConfig,
) -> bool:
    """
    Re-encode a single video to CFR H.264.

    Returns True on success, False on failure.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _get_ffmpeg(),
        "-y",  # overwrite
        "-i",
        str(input_path),
        "-r",
        str(cfg.target_fps),
        "-c:v",
        "libx264",
        "-crf",
        str(cfg.crf_quality),
        "-preset",
        cfg.ffmpeg_preset,
        "-pix_fmt",
        "yuv420p",
        "-an",  # drop audio
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def normalize_all_videos(cfg: PipelineConfig, progress_cb=None) -> dict:
    """
    Normalize every video referenced in labels.csv.

    Args:
        cfg: Pipeline configuration.
        progress_cb: Optional callback(current, total, video_name).

    Returns:
        Dict mapping original relative path → normalized path, plus a
        "failed" key listing any videos that could not be processed.
    """
    import csv

    labels_path = cfg.labels_csv
    results: dict = {"mapping": {}, "failed": []}

    rows = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)

    for idx, row in enumerate(rows):
        rel_path = row["path"]
        src = cfg.dataset_dir / rel_path
        dst = cfg.normalized_video_dir / rel_path

        if dst.exists():
            results["mapping"][rel_path] = dst
            if progress_cb:
                progress_cb(idx + 1, total, rel_path)
            continue

        if not src.exists():
            results["failed"].append((rel_path, "source not found"))
            if progress_cb:
                progress_cb(idx + 1, total, rel_path)
            continue

        ok = normalize_video(src, dst, cfg)
        if ok:
            results["mapping"][rel_path] = dst
        else:
            results["failed"].append((rel_path, "ffmpeg error"))

        if progress_cb:
            progress_cb(idx + 1, total, rel_path)

    return results
