"""
WLASL Dataset Preparation Tool — Main Entry Point

Launch with:
    uv run streamlit run main.py
"""

import streamlit as st

from app.views import (
    keyframe_page,
    landmark_page,
    manual_keyframe_page,
    preprocessing_page,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="WLASL Dataset Preparation Tool",
    page_icon="🖐️",
    layout="wide",
)

# =============================================================================
# CUSTOM STYLES
# =============================================================================
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# MAIN APP
# =============================================================================
st.title("🖐️ WLASL Dataset Preparation Tool")

tab_preprocess, tab_manual, tab_keyframe, tab_landmark = st.tabs(
    [
        "🔧 Video Preprocessing",
        "🖱️ Manual Keyframe Selection",
        "📹 Keyframe Extractor",
        "🦴 Landmark Extractor",
    ]
)

with tab_preprocess:
    preprocessing_page.render()

with tab_manual:
    manual_keyframe_page.render()

with tab_keyframe:
    keyframe_page.render()

with tab_landmark:
    landmark_page.render()
