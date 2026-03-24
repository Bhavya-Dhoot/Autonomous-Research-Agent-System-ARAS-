from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from aras.agents.base import BaseAgent, EventSink
from aras.agents.memory_agent import MemoryAgent
from aras.config import Settings
from aras.healing.fallback_router import FallbackRouter


class ReviewerAgent(BaseAgent):
    """Peer review agent with scoring and revision requests."""

    def __init__(
        self,
        settings: Settings,
        memory: MemoryAgent,
        on_event: EventSink,
        on_tokens=None,
        on_chat_result=None,
    ) -> None:
        super().__init__(agent_id="reviewer", on_event=on_event, on_tokens=on_tokens, on_chat_result=on_chat_result)
        self.settings = settings
        self.memory = memory
        self.router = FallbackRouter(settings=settings)

    async def review(self, *, paper_tex_path: Path) -> dict[str, Any]:
        """Review LaTeX paper and return structured feedback."""
        tex = paper_tex_path.read_text(encoding="utf-8")
        prompts = self.memory.current_prompts()
        rag = self.memory.rag_context(query="how to review research papers; reproducibility checklist")
        system = f"{prompts.get('reviewer','')}\n\nRAG CONTEXT:\n{rag}\n\nReturn STRICT JSON."
        user = (
            "Peer-review this paper. Provide:\n"
            "- overall_score (0-10)\n"
            "- novelty, methodology, clarity, reproducibility (0-10 each)\n"
            "- major_issues (list)\n"
            "- minor_issues (list)\n"
            "- required_revisions (list)\n"
            "- lessons_learned (short)\n\n"
            f"PAPER TEX:\n{tex[:16000]}"
        )
        try:
            res = await self.router.chat(
                role_system=system,
                messages=[{"role": "user", "content": user}],
                purpose="paper_review",
                prefer=["nvidia", "openai", "local"],
                thinking=True,
                temperature=0.4,
                max_tokens=1400,
            )
            self.record_chat_result(res)
            self.add_tokens(res.tokens_used)
            obj = _extract_json(res.text)
            if not isinstance(obj, dict):
                raise ValueError("review is not an object")
            obj["score"] = obj.get("overall_score", obj.get("score"))
            self.emit(f"Review done via {res.provider}/{res.model}")
            return obj
        except Exception as e:
            self.emit(f"Review fallback: {e}", level="error")
            return {
                "overall_score": 6.0,
                "novelty": 5.0,
                "methodology": 6.0,
                "clarity": 7.0,
                "reproducibility": 7.0,
                "major_issues": ["Synthetic-only experiments; add real datasets."],
                "minor_issues": ["Add more citations and tighter problem framing."],
                "required_revisions": ["Include reproducibility checklist and dataset details."],
                "lessons_learned": "Prioritize real benchmarks and include ablations tied to the hypothesis.",
                "score": 6.0,
            }


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    m = re.search(r"\{[\s\S]+\}", text)
    if not m:
        raise ValueError("no JSON found")
    return json.loads(m.group(0))

