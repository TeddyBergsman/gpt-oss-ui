# config.py
import base64
try:
    # Streamlit is optional for desktop usage. If unavailable, fall back gracefully.
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - optional dependency for desktop app
    st = None

# --- Constants ---
MODEL_NAME = "gpt-oss:20b"
ASSISTANT_AVATAR_PATH = "assets/openai.png"
PROMPT_FILE_PATH = "compliance_prompt.txt"
NOTHINK_FILE_PATH = "nothink_prompt.txt"

# --- SVG for User Avatar ---
USER_AVATAR_SVG = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="#000000" stroke="none">
    <circle cx="12" cy="12" r="12"/>
</svg>
"""
USER_AVATAR = f"data:image/svg+xml;base64,{base64.b64encode(USER_AVATAR_SVG.encode('utf-8')).decode('utf-8')}"


def _load_prompt_from_file(file_path):
    """A private helper function to load the prompt text from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # If running under Streamlit, show an error and stop the app.
        if st is not None:
            st.error(f"Fatal Error: The prompt file was not found at '{file_path}'.")
            st.stop()
        # In desktop/CLI contexts, re-raise the error to be handled by the caller.
        raise

# --- Load Content ---
# Load the prompt into a constant. Other modules can import this variable directly.
COMPLIANCE_PROMPT = _load_prompt_from_file(PROMPT_FILE_PATH)
NOTHINK_PROMPT = _load_prompt_from_file(NOTHINK_FILE_PATH)