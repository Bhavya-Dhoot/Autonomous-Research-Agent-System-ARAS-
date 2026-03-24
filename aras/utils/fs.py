from __future__ import annotations

from pathlib import Path


def ensure_dirs(paths: list[Path]) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def safe_write_text(path: Path, content: str) -> None:
    """Write text content atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)

