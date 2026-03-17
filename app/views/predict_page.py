"""
Prediction page for upload-and-classify sign recognition.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from app.core.inference import DEFAULT_CLASSES, predict_video
from app.preprocessing.config import PipelineConfig
from app.preprocessing.video_normalizer import check_ffmpeg


_PREFIX = "pred_"


def _key(name: str) -> str:
    return f"{_PREFIX}{name}"


def _init() -> None:
    defaults = {
        _key("checkpoint_path"): "",
        _key("manual_labels"): ", ".join(DEFAULT_CLASSES),
        _key("threshold"): 0.55,
        _key("result"): None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render() -> None:
    _init()

    st.caption(
        "Upload a sign-language video, run the preprocessing pipeline, and classify it "
        "with a trained checkpoint. This is a closed-vocabulary predictor: it only "
        "recognizes labels the model was trained on."
    )

    if not check_ffmpeg():
        st.error(
            "ffmpeg is not available. Run `uv sync` or install ffmpeg so uploaded videos "
            "can be normalized before inference."
        )
        return

    video_file = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "avi", "mkv"],
        key=_key("video"),
    )
    checkpoint_path = st.text_input(
        "Checkpoint path (.pth)",
        key=_key("checkpoint_path"),
        placeholder="checkpoints/my_model/best_model.pth",
    )

    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state[_key("threshold")]),
            step=0.01,
            key=_key("threshold"),
            help="Below this score the app will say it is not confident.",
        )
    with col2:
        target_length = st.number_input(
            "Target sequence length",
            min_value=8,
            max_value=128,
            value=32,
            step=1,
            key=_key("seq_len"),
        )

    manual_labels = st.text_area(
        "Class labels (comma-separated)",
        key=_key("manual_labels"),
        help="Leave the default 15 labels, or replace them with the exact class order used during training.",
        height=100,
    )

    st.info(
        "Use the exact label order from training. If the checkpoint was trained on a different "
        "class order, predictions will be wrong even if the model loads correctly."
    )

    if st.button("Predict Sign", type="primary", key=_key("run")):
        if video_file is None:
            st.error("Upload a video before running prediction.")
            return
        if not checkpoint_path.strip():
            st.error("Provide a checkpoint path before running prediction.")
            return

        resolved_checkpoint = Path(checkpoint_path).expanduser()
        if not resolved_checkpoint.exists():
            st.error(f"Checkpoint not found: `{resolved_checkpoint}`")
            return

        suffix = Path(video_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_file.read())
            tmp_path = Path(tmp.name)

        cfg = PipelineConfig()

        try:
            with st.spinner("Preprocessing video and running prediction…"):
                result = predict_video(
                    tmp_path,
                    resolved_checkpoint,
                    manual_labels=manual_labels,
                    confidence_threshold=float(threshold),
                    cfg=cfg,
                    target_sequence_length=int(target_length),
                )
            st.session_state[_key("result")] = result
        except Exception as exc:
            st.session_state[_key("result")] = None
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.get(_key("result"))
    if result is not None:
        _render_result(result)


def _render_result(result) -> None:
    st.divider()
    st.subheader("Prediction")

    c1, c2, c3 = st.columns(3)
    c1.metric("Top label", result.predicted_label)
    c2.metric("Confidence", f"{result.confidence:.2%}")
    c3.metric("Status", "Confident" if result.is_confident else "Not confident")

    if result.is_confident:
        st.success(f"Predicted sign: {result.predicted_label}")
    else:
        st.warning(
            f"Top guess is {result.predicted_label}, but confidence is below the threshold."
        )

    st.markdown("**Top predictions**")
    st.dataframe(result.top_k, use_container_width=True, hide_index=True)

    with st.expander("Preprocessing metadata"):
        st.json(result.metadata)