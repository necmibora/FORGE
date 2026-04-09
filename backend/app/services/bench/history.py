from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.schemas import BenchHistoryEntry


class BenchHistoryStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def list(self) -> list[BenchHistoryEntry]:
        if not self._path.exists():
            return []

        entries: list[BenchHistoryEntry] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(BenchHistoryEntry.model_validate_json(line))
                except Exception:
                    continue

        return sorted(
            entries,
            key=lambda item: (
                item.finished_at or item.started_at or 0,
                item.started_at or 0,
            ),
            reverse=True,
        )

    def append(self, entry: BenchHistoryEntry) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n")


history_store = BenchHistoryStore(settings.benchmark_history_path)
