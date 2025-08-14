from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import config


class PresetManager:
    """Manage load/save of UI presets (model, prompt, and settings)."""

    def __init__(self, file_path: Optional[str] = None) -> None:
        base_dir = Path(config.__file__).resolve().parent
        self.file_path = Path(file_path) if file_path else (base_dir / "presets.json")
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.file_path.exists():
            default_preset = {
                "name": "Default",
                "model_name": config.AVAILABLE_MODELS[0]["name"],
                "system_prompt_name": "Assistant",
                "reasoning_effort": "None",
                "temperature": config.AVAILABLE_MODELS[0]["default_temperature"],
                "multi_shot_count": 10,
                "compliance_enabled": False,
            }
            data = {"presets": [default_preset], "last_selected": "Default"}
            self._write_json(data)

    def _read_json(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"presets": [], "last_selected": None}

    def _write_json(self, data: Dict[str, Any]) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def list_presets(self) -> List[Dict[str, Any]]:
        return list(self._read_json().get("presets", []))

    def get_preset_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for p in self.list_presets():
            if p.get("name") == name:
                return p
        return None

    def upsert_preset(self, preset: Dict[str, Any]) -> None:
        data = self._read_json()
        presets = data.get("presets", [])
        existing_index = next((i for i, p in enumerate(presets) if p.get("name") == preset.get("name")), None)
        if existing_index is None:
            presets.append(preset)
        else:
            presets[existing_index] = preset
        data["presets"] = presets
        # If there is no last_selected, set it to this preset
        if not data.get("last_selected"):
            data["last_selected"] = preset.get("name")
        self._write_json(data)

    def delete_preset(self, name: str) -> None:
        data = self._read_json()
        presets = [p for p in data.get("presets", []) if p.get("name") != name]
        data["presets"] = presets
        if data.get("last_selected") == name:
            data["last_selected"] = presets[0]["name"] if presets else None
        self._write_json(data)

    def rename_preset(self, old_name: str, new_name: str) -> None:
        data = self._read_json()
        presets = data.get("presets", [])
        for p in presets:
            if p.get("name") == old_name:
                p["name"] = new_name
                break
        data["presets"] = presets
        if data.get("last_selected") == old_name:
            data["last_selected"] = new_name
        self._write_json(data)

    def get_last_selected(self) -> Optional[str]:
        return self._read_json().get("last_selected")

    def set_last_selected(self, name: str) -> None:
        data = self._read_json()
        data["last_selected"] = name
        self._write_json(data)


