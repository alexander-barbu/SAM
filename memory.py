from __future__ import annotations

import json
import re
from pathlib import Path

MEMORY_FILE = Path("user_memory.json")
_PATTERN = re.compile(r'\[REMEMBER:\s*([^|\]]+?)\s*\|\s*([^\]]+?)\s*\]', re.IGNORECASE)


class MemoryManager:
    def __init__(self, path: Path = MEMORY_FILE):
        self._path = path
        self.facts: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self.facts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def set_fact(self, key: str, value: str) -> None:
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if key and value:
            self.facts[key] = value
            self._save()

    def extract_and_save(self, text: str) -> None:
        for key, value in _PATTERN.findall(text):
            self.set_fact(key, value)

    def as_context_block(self) -> str | None:
        if not self.facts:
            return None
        lines = "\n".join(f"- {k}: {v}" for k, v in self.facts.items())
        return f"What you know about this user:\n{lines}"
