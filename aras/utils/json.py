from __future__ import annotations

import json
from typing import Any

import orjson


def dumps(obj: Any, *, indent: int | None = None) -> str:
    """Serialize to JSON string."""
    if indent is None:
        return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS).decode("utf-8")
    return json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=True)


def loads(s: str) -> Any:
    """Deserialize from JSON string."""
    return json.loads(s)

