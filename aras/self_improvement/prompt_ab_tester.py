from __future__ import annotations

from typing import Any, Literal

from aras.agents.base import BaseAgent, EventSink
from aras.config import Settings
from aras.healing.fallback_router import FallbackRouter
from aras.utils.logging import get_logger


log = get_logger("prompt-ab-tester")


class PromptABTester(BaseAgent):
    """Run an A/B test by generating outputs with two system prompts."""

    def __init__(
        self,
        settings: Settings,
        on_event: EventSink,
        on_tokens=None,
        on_chat_result=None,
    ) -> None:
        super().__init__(agent_id="ab_tester", on_event=on_event, on_tokens=on_tokens, on_chat_result=on_chat_result)
        self.settings = settings
        self.router = FallbackRouter(settings=settings)

    async def test_writer_abstract(
        self,
        *,
        topic: str,
        plan: dict[str, Any],
        system_a: str,
        system_b: str,
    ) -> dict[str, Any]:
        """Compare writer prompts A/B on an abstract-writing microtask."""
        user = (
            "Write an IEEE-style abstract (<=250 words) for the following research plan.\n\n"
            f"TOPIC: {topic}\n"
            f"HYPOTHESIS: {plan.get('hypothesis')}\n"
            f"QUESTIONS: {plan.get('questions')}\n"
            f"METRICS: {plan.get('metrics')}\n"
        )
        res_a = await self.router.chat(
            role_system=system_a,
            messages=[{"role": "user", "content": user}],
            purpose="ab_test_writer_abstract_a",
            prefer=["nvidia", "openai", "local"],
            thinking=False,
            temperature=0.4,
            max_tokens=450,
        )
        self.record_chat_result(res_a)

        res_b = await self.router.chat(
            role_system=system_b,
            messages=[{"role": "user", "content": user}],
            purpose="ab_test_writer_abstract_b",
            prefer=["nvidia", "openai", "local"],
            thinking=False,
            temperature=0.4,
            max_tokens=450,
        )
        self.record_chat_result(res_b)

        a_text = res_a.text.strip()
        b_text = res_b.text.strip()

        score_a = self._heuristic_score_abstract(a_text)
        score_b = self._heuristic_score_abstract(b_text)
        winner: Literal["a", "b"] = "a" if score_a >= score_b else "b"
        return {
            "winner": winner,
            "score_a": float(score_a),
            "score_b": float(score_b),
            "abstract_a_preview": a_text[:500],
            "abstract_b_preview": b_text[:500],
        }

    def _heuristic_score_abstract(self, text: str) -> float:
        """Heuristic score in [0,1] based on length and presence of typical abstract signals."""
        t = (text or "").strip()
        if not t:
            return 0.0
        words = [w for w in t.replace("\n", " ").split(" ") if w.strip()]
        n = len(words)
        # Prefer around 160-230 words.
        if n <= 0:
            return 0.0
        length_score = 1.0 - min(1.0, abs(n - 200) / 120.0)
        low = t.lower()
        has_we = 1.0 if "we " in low or low.startswith("we") else 0.0
        has_contribution = 1.0 if any(k in low for k in ["we propose", "we present", "this paper", "we study", "we introduce"]) else 0.0
        # Weighted blend.
        score = 0.65 * length_score + 0.2 * has_we + 0.15 * has_contribution
        return float(max(0.0, min(1.0, score)))

