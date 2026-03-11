"""
Manual Keyframe Selection Page

Select a video from the dataset → frames are extracted using ffmpeg (same
pipeline as the preprocessor for consistent frame counts) → click thumbnails
to toggle selection → save as ground-truth data for evaluating automatic
keyframe extraction algorithms.

Ground truth is saved as:
  dataset/ground_truth/<label>/<video_stem>/
      frames/              ← JPEG images of selected keyframes
      ground_truth.json    ← metadata + selected indices
      ground_truth.npy     ← numpy int32 array of selected indices
"""

import csv
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from app.preprocessing.video_normalizer import _get_ffmpeg, check_ffmpeg


# ── session-state keys ───────────────────────────────────────────────────────
_ALL_FRAMES = "mk_all_frames"
_SELECTED = "mk_selected_indices"  # set[int]
_VIDEO_NAME = "mk_video_name"
_VIDEO_LABEL = "mk_video_label"
_VIDEO_META = "mk_video_meta"
_LOAD_KEY = "mk_load_key"
_PAGE = "mk_page"  # current pagination page

_DATASET_DIR = Path("dataset/raw_video_data")
_LABELS_CSV = _DATASET_DIR / "labels.csv"
_GT_ROOT = Path("dataset/ground_truth")
_TARGET_FPS = 25  # must match PipelineConfig.target_fps
_FRAMES_PER_PAGE = 18  # 3 rows × 6 cols — keeps the grid fast
_THUMB_MAX = 160  # max thumbnail width/height for faster rendering


def _init():
    defaults = {
        _ALL_FRAMES: None,
        _SELECTED: set(),
        _VIDEO_NAME: "",
        _VIDEO_LABEL: "",
        _VIDEO_META: {},
        _LOAD_KEY: "",
        _PAGE: 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── ffmpeg-based frame extraction ────────────────────────────────────────────


def _extract_frames_ffmpeg(
    video_path: Path, fps: int = _TARGET_FPS
) -> list[np.ndarray]:
    """
    Extract all frames from a video using ffmpeg at a constant frame rate.

    This ensures frame counts match the preprocessing pipeline exactly.
    Returns a list of RGB uint8 numpy arrays.
    """
    ffmpeg = _get_ffmpeg()

    with tempfile.TemporaryDirectory() as tmp_dir:
        pattern = str(Path(tmp_dir) / "frame_%06d.png")

        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-r",
            str(fps),
            "-f",
            "image2",
            "-pix_fmt",
            "rgb24",
            pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg frame extraction failed:\n{result.stderr[-500:]}"
            )

        # Read frames in sorted order
        frame_files = sorted(Path(tmp_dir).glob("frame_*.png"))
        frames = []
        for fp in frame_files:
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return frames


def _get_video_meta_ffmpeg(video_path: Path) -> dict:
    """Probe video metadata using ffmpeg."""
    ffmpeg = _get_ffmpeg()
    # Use ffmpeg's stderr output to get stream info
    cmd = [ffmpeg, "-i", str(video_path), "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    stderr = result.stderr

    # Parse basic info from stderr
    meta = {"native_fps": 0.0, "width": 0, "height": 0, "duration": 0.0}

    import re

    # Match "Stream ... Video: ... 1920x1080 ... 24 fps"
    stream_match = re.search(r"(\d{2,5})x(\d{2,5})", stderr)
    if stream_match:
        meta["width"] = int(stream_match.group(1))
        meta["height"] = int(stream_match.group(2))

    fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", stderr)
    if fps_match:
        meta["native_fps"] = float(fps_match.group(1))

    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", stderr)
    if duration_match:
        h, m, s, cs = duration_match.groups()
        meta["duration"] = int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100

    return meta


# ── ground truth save ────────────────────────────────────────────────────────


def _save_ground_truth() -> tuple[bool, str]:
    frames: list = st.session_state[_ALL_FRAMES]
    sel_indices = sorted(st.session_state[_SELECTED])
    video_name: str = st.session_state[_VIDEO_NAME]
    label: str = st.session_state[_VIDEO_LABEL]
    meta: dict = st.session_state[_VIDEO_META]

    if not sel_indices:
        return False, "No frames selected."

    video_stem = Path(video_name).stem
    save_root = _GT_ROOT / label / video_stem
    frames_dir = save_root / "frames"

    try:
        frames_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"Cannot create directory: {e}"

    # Save selected frame JPEGs
    for idx in sel_indices:
        bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frames_dir / f"frame_{idx:05d}.jpg"), bgr)

    # Save JSON manifest
    manifest = {
        "video": video_name,
        "label": label,
        "target_fps": _TARGET_FPS,
        "native_fps": meta.get("native_fps", 0),
        "native_resolution": f"{meta.get('width', 0)}x{meta.get('height', 0)}",
        "total_frames": len(frames),
        "selected_count": len(sel_indices),
        "selected_indices": sel_indices,
    }
    with open(save_root / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save numpy array
    np.save(str(save_root / "ground_truth.npy"), np.array(sel_indices, dtype=np.int32))

    return True, str(save_root)


# ── read labels.csv ──────────────────────────────────────────────────────────


def _read_labels() -> list[dict]:
    if not _LABELS_CSV.exists():
        return []
    with open(_LABELS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_thumbnail(frame: np.ndarray, selected: bool) -> np.ndarray:
    """Downscale frame for fast display and optionally draw selection border."""
    h, w = frame.shape[:2]
    scale = _THUMB_MAX / max(h, w)
    if scale < 1.0:
        thumb = cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        thumb = frame

    if selected:
        thumb = thumb.copy()
        border = max(3, thumb.shape[0] // 25)
        cv2.rectangle(
            thumb,
            (0, 0),
            (thumb.shape[1] - 1, thumb.shape[0] - 1),
            (50, 205, 50),
            thickness=border,
        )
    return thumb


def _toggle_frame(idx: int):
    """on_click callback — toggles frame selection without explicit rerun."""
    sel: set = st.session_state[_SELECTED]
    if idx in sel:
        sel.discard(idx)
    else:
        sel.add(idx)


def _set_page(p: int):
    """on_click callback for pagination."""
    st.session_state[_PAGE] = p


def _render_frame_grid(frames: list[np.ndarray]):
    """Paginated frame grid.  Uses on_click callbacks for zero-lag toggling."""
    selected: set = st.session_state[_SELECTED]
    total = len(frames)
    total_pages = max(1, (total + _FRAMES_PER_PAGE - 1) // _FRAMES_PER_PAGE)

    # Clamp page
    page = st.session_state.get(_PAGE, 0)
    page = max(0, min(page, total_pages - 1))
    st.session_state[_PAGE] = page

    st.markdown("---")
    st.markdown("### 🎞️ Click a frame to toggle selection")
    st.caption("🟩 Green border = selected  |  Click again to deselect")

    # Pagination controls
    pg_left, pg_info, pg_right = st.columns([1, 3, 1])
    start_idx = page * _FRAMES_PER_PAGE
    end_idx = min(start_idx + _FRAMES_PER_PAGE, total)
    with pg_left:
        st.button(
            "◀ Prev",
            disabled=page == 0,
            key="mk_pg_prev",
            on_click=_set_page,
            args=(page - 1,),
        )
    with pg_info:
        st.markdown(
            f"**Page {page + 1}/{total_pages}**  —  "
            f"frames {start_idx}–{end_idx - 1}  |  "
            f"**{len(selected)}** selected"
        )
    with pg_right:
        st.button(
            "Next ▶",
            disabled=page >= total_pages - 1,
            key="mk_pg_next",
            on_click=_set_page,
            args=(page + 1,),
        )

    # Render only current page of frames
    COLS = 6
    page_indices = list(range(start_idx, end_idx))
    num_rows = (len(page_indices) + COLS - 1) // COLS

    for row_idx in range(num_rows):
        cols = st.columns(COLS)
        for col_idx, col in enumerate(cols):
            i = row_idx * COLS + col_idx
            if i >= len(page_indices):
                break
            frame_idx = page_indices[i]
            is_selected = frame_idx in selected
            with col:
                thumb = _make_thumbnail(frames[frame_idx], is_selected)
                cap = f"{'✅ ' if is_selected else ''}#{frame_idx}"
                st.image(thumb, caption=cap, use_container_width=True)
                st.button(
                    "Deselect" if is_selected else "Select",
                    key=f"mk_btn_{frame_idx}",
                    type="primary" if is_selected else "secondary",
                    on_click=_toggle_frame,
                    args=(frame_idx,),
                )


# ── main render ──────────────────────────────────────────────────────────────


def render():
    _init()

    st.caption(
        "Select keyframes manually to build ground-truth data for evaluating "
        "automatic extraction algorithms. Frames are extracted at "
        f"{_TARGET_FPS} fps using ffmpeg — matching the preprocessing pipeline."
    )

    # --- ffmpeg check ---
    if not check_ffmpeg():
        st.error(
            "**ffmpeg is not available.** "
            "Run `uv sync` to install the bundled ffmpeg binary."
        )
        return

    # ── 1. VIDEO SOURCE ───────────────────────────────────────────────────────
    source = st.radio(
        "Video source",
        ["Select from dataset", "Upload file"],
        horizontal=True,
        key="mk_source",
    )

    video_path: Path | None = None
    video_name = ""
    label = ""

    if source == "Select from dataset":
        rows = _read_labels()
        if not rows:
            st.warning(f"No labels.csv found at `{_LABELS_CSV}`")
            return

        options = [f"{r['label']}/{r['video_name']}" for r in rows]
        choice = st.selectbox("Select video", options, key="mk_video_sel")
        idx = options.index(choice)
        row = rows[idx]

        video_path = _DATASET_DIR / row["path"]
        video_name = row["video_name"]
        label = row["label"]

        if not video_path.exists():
            st.error(f"File not found: `{video_path}`")
            return
    else:
        uploaded = st.file_uploader(
            "Upload Video", type=["mp4", "mov", "avi", "mkv"], key="mk_uploader"
        )
        if uploaded is None:
            st.info("Upload a video to get started.")
            return

        # Save uploaded file to temp location
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()
        video_path = Path(tmp.name)
        video_name = uploaded.name
        label = st.text_input("Label (class name)", value="unknown", key="mk_label")

    # ── 2. LOAD FRAMES (only when video changes) ─────────────────────────────
    load_key = f"{label}/{video_name}"

    if st.session_state[_LOAD_KEY] != load_key:
        with st.spinner(f"Extracting frames at {_TARGET_FPS} fps using ffmpeg…"):
            meta = _get_video_meta_ffmpeg(video_path)
            frames = _extract_frames_ffmpeg(video_path, fps=_TARGET_FPS)

        st.session_state[_ALL_FRAMES] = frames
        st.session_state[_SELECTED] = set()
        st.session_state[_VIDEO_NAME] = video_name
        st.session_state[_VIDEO_LABEL] = label
        st.session_state[_VIDEO_META] = meta
        st.session_state[_LOAD_KEY] = load_key

    frames: list = st.session_state[_ALL_FRAMES]
    meta: dict = st.session_state[_VIDEO_META]
    selected: set = st.session_state[_SELECTED]

    if not frames:
        st.error("Could not extract any frames from this video.")
        return

    # ── 3. VIDEO SUMMARY ─────────────────────────────────────────────────────
    h, w = frames[0].shape[:2]
    st.caption(
        f"📹 **{video_name}**  |  "
        f"native {meta.get('width', '?')}×{meta.get('height', '?')} "
        f"@ {meta.get('native_fps', '?')} fps  →  "
        f"**{len(frames)} frames** extracted @ {_TARGET_FPS} fps ({w}×{h})"
    )

    # Check if ground truth already exists
    existing_gt = _GT_ROOT / label / Path(video_name).stem / "ground_truth.json"
    if existing_gt.exists():
        with open(existing_gt, encoding="utf-8") as f:
            existing = json.load(f)
        st.info(
            f"ℹ️ Ground truth already exists for this video "
            f"({existing['selected_count']} frames selected). "
            f"Saving again will overwrite it."
        )
        if st.button("📂 Load existing selection", key="mk_load_existing"):
            st.session_state[_SELECTED] = set(existing["selected_indices"])
            st.rerun()

    # ── 4. QUICK SELECTION HELPERS ────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Select All"):
            st.session_state[_SELECTED] = set(range(len(frames)))
            st.rerun()
    with col_b:
        if st.button("❌ Clear All"):
            st.session_state[_SELECTED] = set()
            st.rerun()

    selected_count = len(st.session_state[_SELECTED])
    st.markdown(f"**{selected_count} / {len(frames)} frames selected**")

    # ── 5. FRAME GRID (fragment — only this section re-runs on click) ─────────
    _render_frame_grid(frames)

    # ── 6. SAVE GROUND TRUTH ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Save Ground Truth")

    save_path = _GT_ROOT / label / Path(video_name).stem
    st.caption(f"Will save to: `{save_path}/`")

    # Re-read selected count (may have changed inside fragment)
    selected_count = len(st.session_state[_SELECTED])

    if selected_count == 0:
        st.warning("Select at least one frame before saving.")
    else:
        st.success(
            f"**{selected_count} frames** ready to save: "
            f"{sorted(st.session_state[_SELECTED])[:10]}{'…' if selected_count > 10 else ''}"
        )

    if st.button("💾 Save Ground Truth", type="primary", disabled=selected_count == 0):
        ok, msg = _save_ground_truth()
        if ok:
            st.success(f"✅ Saved to `{msg}`")
            st.markdown(
                f"- `{msg}/frames/frame_XXXXX.jpg` × {selected_count}\n"
                f"- `{msg}/ground_truth.json`\n"
                f"- `{msg}/ground_truth.npy`"
            )
        else:
            st.error(msg)

    # ── 7. PREVIEW SELECTED FRAMES ────────────────────────────────────────────
    if selected_count > 0:
        with st.expander(
            f"🔍 Preview {selected_count} selected frames", expanded=False
        ):
            sel_sorted = sorted(st.session_state[_SELECTED])
            prev_cols = st.columns(min(len(sel_sorted), 5))
            for i, idx in enumerate(sel_sorted):
                with prev_cols[i % 5]:
                    st.image(
                        frames[idx],
                        caption=f"Frame #{idx}",
                        use_container_width=True,
                    )
