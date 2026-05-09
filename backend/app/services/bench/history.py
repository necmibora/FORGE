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

    def delete(self, ids: set[str]) -> int:
        """Remove entries whose id is in `ids`. Returns number of rows deleted.

        Reads the JSONL, filters, and rewrites atomically via a temp file.
        """
        if not ids or not self._path.exists():
            return 0

        kept: list[str] = []
        deleted = 0
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                    if obj.get("id") in ids:
                        deleted += 1
                        continue
                except Exception:
                    pass
                kept.append(stripped)

        if deleted == 0:
            return 0

        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")
        tmp.replace(self._path)
        return deleted


history_store = BenchHistoryStore(settings.benchmark_history_path)
