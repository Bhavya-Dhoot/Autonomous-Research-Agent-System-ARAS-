from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler


def configure_logging(logs_dir: Path) -> None:
    """Configure console + file logging."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_path = logs_dir / "aras.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    console = RichHandler(rich_tracebacks=True, show_time=False, show_level=True, show_path=False)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    root.addHandler(console)
    root.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    sys.excepthook = _excepthook


def _excepthook(exc_type: type[BaseException], exc: BaseException, tb) -> None:  # type: ignore[no-untyped-def]
    logging.getLogger("uncaught").exception("Uncaught exception", exc_info=(exc_type, exc, tb))


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)

