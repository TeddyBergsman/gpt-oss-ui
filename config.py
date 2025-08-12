# config.py
import base64
from pathlib import Path

# --- Constants ---
MODEL_NAME = "gemma3:12b"  # Default model

# Available models with their capabilities
AVAILABLE_MODELS = [
    {
        "name": "gemma3:12b",
        "display_name": "Gemma3 12B",
        "supports_reasoning": False,
        "supports_compliance": False,
        "supports_images": True,
        "context_window": 128000,  # 128K tokens
    },
    {
        "name": "gpt-oss:20b",
        "display_name": "GPT-OSS 20B",
        "supports_reasoning": True,
        "supports_compliance": True,
        "supports_images": False,
        "context_window": 128000,  # Estimated, no official spec found
    },
    {
        "name": "gpt-oss:120b",
        "display_name": "GPT-OSS 120B",
        "supports_reasoning": True,
        "supports_compliance": True,
        "supports_images": False,
        "context_window": 128000,  # Estimated, no official spec found
    },
    {
        "name": "gemma3:27b",
        "display_name": "Gemma3 27B",
        "supports_reasoning": False,
        "supports_compliance": False,
        "supports_images": True,
        "context_window": 128000,  # 128K tokens
    },
    {
        "name": "gemma3:4b",
        "display_name": "Gemma3 4B",
        "supports_reasoning": False,
        "supports_compliance": False,
        "supports_images": True,
        "context_window": 128000,  # 128K tokens
    },
    {
        "name": "gemma3:1b",
        "display_name": "Gemma3 1B",
        "supports_reasoning": False,
        "supports_compliance": False,
        "supports_images": True,
        "context_window": 32000,  # 32K tokens
    },
    {
        "name": "qwen3:30b",
        "display_name": "Qwen3 30B",
        "supports_reasoning": True,  # Has thinking/non-thinking modes
        "supports_compliance": False,
        "supports_images": False,
        "context_window": 256000,  # Up to 131K with YaRN
    },
    {
        "name": "qwen3:235b",
        "display_name": "Qwen3 235B",
        "supports_reasoning": True,  # Has thinking/non-thinking modes
        "supports_compliance": False,
        "supports_images": False,
        "context_window": 256000, 
    },
    {
        "name": "deepseek-r1:8b",
        "display_name": "DeepSeek-R1 8B",
        "supports_reasoning": True,  # R1 series has reasoning capabilities
        "supports_compliance": False,
        "supports_images": False,
        "context_window": 128000,  # Estimated, no official spec found
    },
    {
        "name": "deepseek-r1:32b",
        "display_name": "DeepSeek-R1 32B",
        "supports_reasoning": True,  # R1 series has reasoning capabilities
        "supports_compliance": False,
        "supports_images": False,
        "context_window": 128000,  # Estimated, no official spec found
    },
    {
        "name": "deepseek-r1:70b",
        "display_name": "DeepSeek-R1 70B",
        "supports_reasoning": True,  # R1 series has reasoning capabilities
        "supports_compliance": False,
        "supports_images": False,
        "context_window": 128000,  # Estimated, no official spec found
    },
    {
        "name": "huihui_ai/gemma3n-abliterated:e2b-fp16",
        "display_name": "Gemma3n 2B (Abliterated)",
        "supports_reasoning": False,
        "supports_compliance": False,  # Uncensored models don't support compliance
        "supports_images": True,  # Based on Gemma3
        "context_window": 32000,  # Based on Gemma3
    },
    {
        "name": "huihui_ai/gemma3n-abliterated:e4b-fp16",
        "display_name": "Gemma3n 4B (Abliterated)",
        "supports_reasoning": False,
        "supports_compliance": False,  # Uncensored models don't support compliance
        "supports_images": True,  # Based on Gemma3
        "context_window": 32000,  # Based on Gemma3
    },
    {
        "name": "huihui_ai/mistral-small-abliterated:24b",
        "display_name": "Mistral Small 24B (Abliterated)",
        "supports_reasoning": False,
        "supports_compliance": False,  # Uncensored models don't support compliance
        "supports_images": False,
        "context_window": 32000,  # Based on typical Mistral models
    },
    {
        "name": "redule26/huihui_ai_qwen2.5-vl-7b-abliterated:latest",
        "display_name": "Qwen2.5-VL 7B (Abliterated)",
        "supports_reasoning": False,
        "supports_compliance": False,  # Uncensored models don't support compliance
        "supports_images": True,  # VL = Vision-Language model
        "context_window": 125000,  # Based on typical Qwen2.5 models
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