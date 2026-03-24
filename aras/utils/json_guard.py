from __future__ import annotations

import json
from typing import Any, Callable, TypeVar


T = TypeVar("T")


def safe_json_loads(data: str, default: T | None = None) -> T | None:
    """Safely parse JSON string, returning default on error.

    Parameters
    ----------
    data:
        JSON string to parse.
    default:
        Value to return if parsing fails.
    """
    try:
        return json.loads(data)
    except Exception:
        return default


def ensure_dict(obj: Any) -> dict[str, Any]:
    """Ensure a JSON-like object is a dict[str, Any]."""
    if isinstance(obj, dict):
        return obj
    return {}


def ensure_list(obj: Any) -> list[Any]:
    """Ensure a JSON-like object is a list."""
    if isinstance(obj, list):
        return obj
    return []


def coerce(obj: Any, transform: Callable[[Any], T], default: T) -> T:
    """Apply transform to obj, returning default on any failure."""
    try:
        return transform(obj)
    except Exception:
        return default

