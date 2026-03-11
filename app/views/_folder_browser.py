"""
Cross-platform folder browser dialog (package-private to views).

Uses tkinter's native folder picker, which works on macOS, Windows, and Linux.
"""

import streamlit as st


def _open_folder_dialog(title: str = "Select Folder") -> str:
    """
    Open a native OS folder picker and return the selected path (or "").

    Works on Windows, macOS, and Linux (requires ``python3-tk`` on some
    Linux distributions).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        st.error(
            "tkinter is not available. On Linux, install it with: "
            "`sudo apt-get install python3-tk`"
        )
        return ""

    root = tk.Tk()
    root.withdraw()
    # Bring dialog to front on all platforms
    root.wm_attributes("-topmost", True)
    folder = filedialog.askdirectory(master=root, title=title)
    root.destroy()
    return folder or ""


def folder_input_with_browse(
    label: str,
    session_key: str,
    placeholder: str = "/absolute/path/to/folder",
    dialog_title: str = "Select Folder",
) -> str:
    """
    Render a text input alongside a **Browse** button.

    Returns the current folder path string (may be empty).

    Parameters
    ----------
    label : str
        Label for the text input widget.
    session_key : str
        Streamlit session-state key used to persist the path.
    placeholder : str
        Placeholder text shown when the input is empty.
    dialog_title : str
        Title of the native folder-picker dialog.
    """
    # A separate key stores the result from the browse dialog.  We apply it
    # to the widget key *before* the widget is instantiated so Streamlit
    # doesn't complain about mutating state after widget creation.
    pending_key = f"{session_key}__browse_pending"

    if pending_key in st.session_state and st.session_state[pending_key]:
        st.session_state[session_key] = st.session_state[pending_key]
        del st.session_state[pending_key]

    col_input, col_btn = st.columns([5, 1])

    with col_input:
        folder = st.text_input(
            label,
            key=session_key,
            placeholder=placeholder,
        )

    with col_btn:
        # Add vertical spacing so the button aligns with the text input
        st.markdown("<div style='margin-top: 1.65rem'></div>", unsafe_allow_html=True)
        if st.button("📁 Browse", key=f"{session_key}__browse_btn"):
            result = _open_folder_dialog(title=dialog_title)
            if result:
                # Store in a pending key and rerun; on the next cycle the
                # value is moved into the widget key before it renders.
                st.session_state[pending_key] = result
                st.rerun()

    return folder or ""
