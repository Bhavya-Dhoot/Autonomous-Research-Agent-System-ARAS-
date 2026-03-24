from __future__ import annotations

import json
from typing import Any

from aras.agents.memory_agent import MemoryAgent
from aras.agents.reviewer_agent import ReviewerAgent
from aras.config import Settings
from aras.healing.fallback_router import FallbackRouter


class PromptEvolver:
    """Evolve prompts based on reviewer feedback and store new versions."""

    def __init__(self, settings: Settings, reviewer: ReviewerAgent, memory: MemoryAgent) -> None:
        self.settings = settings
        self.reviewer = reviewer
        self.memory = memory
        self.router = FallbackRouter(settings=settings)

    async def evolve(self, *, topic: str, review: dict[str, Any]) -> dict[str, Any]:
        prompts = self.memory.current_prompts()
        rag = self.memory.rag_context(query=f"prompt improvements for topic {topic}")
        system = (
            "You are a prompt engineer. Update per-agent system prompts to address weaknesses.\n"
            "Return STRICT JSON: {updated_prompts: {agent: prompt}, lessons_learned: str}.\n"
            "Do not remove safety constraints; emphasize reproducibility."
        )
        user = (
            f"TOPIC: {topic}\n\n"
            f"CURRENT PROMPTS:\n{json.dumps(prompts, ensure_ascii=False, indent=2)[:12000]}\n\n"
            f"REVIEW:\n{json.dumps(review, ensure_ascii=False, indent=2)[:8000]}\n\n"
            f"RAG:\n{rag[:4000]}\n"
        )
        updated_prompts = prompts
        lessons = str(review.get("lessons_learned") or "")
        try:
            res = await self.router.chat(
                role_system=system,
                messages=[{"role": "user", "content": user}],
                purpose="prompt_evolution",
                prefer=["nvidia", "openai", "local"],
                thinking=True,
                temperature=0.4,
                max_tokens=1400,
            )
            self.memory.record_chat_result(res)
            obj = _extract_json(res.text)
            if isinstance(obj, dict) and isinstance(obj.get("updated_prompts"), dict):
                updated_prompts = {**prompts, **obj["updated_prompts"]}
            if isinstance(obj, dict) and obj.get("lessons_learned"):
                lessons = str(obj["lessons_learned"])
        except Exception:
            updated_prompts = _heuristic_update(prompts, review)

        version = await self.memory.bump_prompts(updated=updated_prompts)
        return {"prompt_version": version, "lessons_learned": lessons}


def _extract_json(text: str) -> Any:
    import re

    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)
    m = re.search(r"\{[\s\S]+\}", t)
    if not m:
        raise ValueError("no JSON found")
    return json.loads(m.group(0))


def _heuristic_update(prompts: dict[str, str], review: dict[str, Any]) -> dict[str, str]:
    upd = dict(prompts)
    majors = review.get("major_issues") or []
    if majors:
        extra = "\n\nPrioritize addressing major issues: " + "; ".join([str(x) for x in majors[:4]])
        for k in ["writer", "research", "analyst", "scraping", "coder"]:
            upd[k] = (upd.get(k, "") + extra).strip()
    return upd

