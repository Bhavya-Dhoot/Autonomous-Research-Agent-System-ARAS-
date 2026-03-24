from __future__ import annotations

from pathlib import Path
from typing import Any

from aras.agents.reviewer_agent import ReviewerAgent
from aras.config import Settings


class PaperScorer:
    """Compute a paper score from reviewer feedback."""

    def __init__(self, settings: Settings, reviewer: ReviewerAgent) -> None:
        self.settings = settings
        self.reviewer = reviewer

    async def score(self, *, paper_tex_path: Path, review: dict[str, Any]) -> float | None:
        """Return a scalar score (0-10)."""
        try:
            if "overall_score" in review:
                return float(review["overall_score"])
            if "score" in review:
                return float(review["score"])
            comps = [review.get(k) for k in ["novelty", "methodology", "clarity", "reproducibility"]]
            vals = [float(v) for v in comps if v is not None]
            return float(sum(vals) / max(1, len(vals)))
        except Exception:
            return None

