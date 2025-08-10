# config.py
import base64
from pathlib import Path

# --- Constants ---
MODEL_NAME = "gpt-oss:20b"  # Default model

# Available models with their capabilities
AVAILABLE_MODELS = [
    {
        "name": "gpt-oss:20b",
        "display_name": "GPT-OSS 20B",
        "supports_reasoning": True,
        "supports_compliance": True,
        "supports_images": False,
    },
    {
        "name": "gemma3:12b",
        "display_name": "Gemma3 12B",
        "supports_reasoning": False,
        "supports_compliance": False,
        "supports_images": True,
    },
]

# Use absolute paths for robustness
_BASE_DIR = Path(__file__).resolve().parent
ASSISTANT_AVATAR_PATH = str((_BASE_DIR / "assets" / "openai.png").resolve())
PROMPT_FILE_PATH = str((_BASE_DIR / "compliance_prompt.txt").resolve())
NOTHINK_FILE_PATH = str((_BASE_DIR / "nothink_prompt.txt").resolve())

# --- SVG for User Avatar ---
USER_AVATAR_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="#000000" stroke="none">
    <circle cx="12" cy="12" r="12"/>
</svg>
"""
USER_AVATAR = f"data:image/svg+xml;base64,{base64.b64encode(USER_AVATAR_SVG.encode('utf-8')).decode('utf-8')}"


def _load_prompt_from_file(file_path: str) -> str:
    """Load prompt text from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# --- Load Content ---
COMPLIANCE_PROMPT = _load_prompt_from_file(PROMPT_FILE_PATH)
NOTHINK_PROMPT = _load_prompt_from_file(NOTHINK_FILE_PATH)