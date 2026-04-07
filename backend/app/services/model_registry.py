"""Scan MODELS_DIR and QUANT_DIR for HuggingFace-style model directories."""
from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.schemas import ModelEntry


def _is_model_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    return (p / "config.json").is_file()


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _detect_quant(p: Path) -> str | None:
    cfg = p / "config.json"
    if not cfg.is_file():
        return None
    try:
        data = json.loads(cfg.read_text())
    except Exception:
        return None
    qc = data.get("quantization_config")
    if isinstance(qc, dict):
        return qc.get("quant_method") or qc.get("quantization") or "unknown"
    return None


def _scan(root: Path, kind: str) -> list[ModelEntry]:
    if not root.exists():
        return []
    out: list[ModelEntry] = []
    for child in sorted(root.iterdir()):
        if _is_model_dir(child):
            out.append(
                ModelEntry(
                    name=child.name,
                    path=str(child.resolve()),
                    kind=kind,  # type: ignore[arg-type]
                    size_bytes=_dir_size(child),
                    quantization=_detect_quant(child),
                )
            )
    return out


def list_models() -> list[ModelEntry]:
    return _scan(settings.models_dir, "base") + _scan(settings.quant_dir, "quant")
