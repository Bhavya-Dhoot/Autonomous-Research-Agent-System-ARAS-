from __future__ import annotations

import re
from typing import Any


def slugify(value: Any, max_length: int | None = 80) -> str:
    """Convert arbitrary value to a filesystem- and URL-safe slug.

    Parameters
    ----------
    value:
        Input value to slugify; will be converted to string.
    max_length:
        Optional maximum slug length (characters). If None, no limit.
    """
    text = str(value).strip().lower()
    # Replace non-alphanumeric with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if not text:
        text = "topic"
    if max_length is not None and max_length > 0:
        text = text[:max_length].rstrip("-")
    return text or "topic"

