from __future__ import annotations

from typing import Any


def extract_lessons(*, review: dict[str, Any], analysis: dict[str, Any]) -> str:
    """Extract a concise lessons-learned string from review/analysis."""
    for k in ["lessons_learned", "lesson", "lessons"]:
        if review.get(k):
            return str(review[k])
        if analysis.get(k):
            return str(analysis[k])
    majors = review.get("major_issues") or []
    if majors:
        return "Address major issues: " + "; ".join([str(x) for x in majors[:3]])
    return "Improve benchmark coverage, citations, and explicit reproducibility steps."

