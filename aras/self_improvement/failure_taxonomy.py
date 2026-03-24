from __future__ import annotations

import re
from typing import Any


def classify_failure(*, error_text: str) -> dict[str, Any]:
    """Classify an experiment failure into a coarse taxonomy.

    Returns a dict with:
    - failure_type: stable string for downstream targeted healing
    - message: short normalized message
    """
    t = (error_text or "").strip()
    low = t.lower()

    def _short() -> str:
        s = re.sub(r"\s+", " ", t)
        return s[:200]

    if "modulenotfounderror" in low:
        return {"failure_type": "module_not_found", "message": _short()}
    if "importerror" in low:
        return {"failure_type": "import_error", "message": _short()}
    if "syntaxerror" in low:
        return {"failure_type": "syntax_error", "message": _short()}
    if "nameerror" in low:
        return {"failure_type": "name_error", "message": _short()}
    if "valueerror" in low:
        return {"failure_type": "value_error", "message": _short()}
    if "timeout" in low or "timed out" in low:
        return {"failure_type": "timeout", "message": _short()}
    if "nan" in low or "is nan" in low or "overflow" in low:
        return {"failure_type": "numerical_instability", "message": _short()}
    if "runtimeerror" in low:
        return {"failure_type": "runtime_error", "message": _short()}

    return {"failure_type": "unknown_error", "message": _short()}

