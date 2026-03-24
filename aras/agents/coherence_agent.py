from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from aras.agents.base import BaseAgent, EventSink
from aras.agents.memory_agent import MemoryAgent
from aras.config import Settings
from aras.healing.fallback_router import FallbackRouter
from aras.utils.logging import get_logger


log = get_logger("coherence-agent")


class CoherenceAgent(BaseAgent):
    """Revise paper LaTeX for coherence using review feedback."""

    def __init__(
        self,
        settings: Settings,
        memory: MemoryAgent,
        on_event: EventSink,
        on_tokens=None,
        on_chat_result=None,
    ) -> None:
        super().__init__(agent_id="coherence", on_event=on_event, on_tokens=on_tokens, on_chat_result=on_chat_result)
        self.settings = settings
        self.memory = memory
        self.router = FallbackRouter(settings=settings)

    async def revise(
        self,
        *,
        topic: str,
        paper_tex: str,
        review: dict[str, Any],
        round: int,
        max_input_chars: int = 16000,
    ) -> str:
        """Return revised LaTeX document (full tex)."""
        try:
            prompts = self.memory.current_prompts()
            rag = self.memory.rag_context(query=f"coherence improvements for {topic}", collections=["research_memory"], n=4)
            required = review.get("required_revisions") or []

            system = f"{prompts.get('coherence','')}\n\nRAG CONTEXT:\n{rag}\n\nReturn STRICT JSON."
            user = (
                f"Coherence revision round {round}.\n\n"
                f"TOPIC: {topic}\n\n"
                "REVIEW FEEDBACK:\n"
                f"{json.dumps(review, ensure_ascii=False, indent=2)}\n\n"
                f"REQUIRED_REVISIONS:\n{json.dumps(required, ensure_ascii=False)}\n\n"
                "PASTE CURRENT PAPER TEX (may be truncated for context):\n"
                f"{paper_tex[:max_input_chars]}\n\n"
                "Return STRICT JSON with keys:\n"
                "- revised_tex: string (full LaTeX document)\n"
                "- notes: short string\n"
                "Ensure revised_tex remains a complete LaTeX document and preserves BibTeX placeholder/bibliography calls."
            )

            res = await self.router.chat(
                role_system=system,
                messages=[{"role": "user", "content": user}],
                purpose="coherence_revision",
                prefer=["nvidia", "openai", "local"],
                thinking=True,
                temperature=0.2,
                max_tokens=8000,
            )
            self.record_chat_result(res)
            obj = self._extract_json(res.text)
            if isinstance(obj, dict) and isinstance(obj.get("revised_tex"), str):
                return obj["revised_tex"]
        except Exception as e:
            self.emit(f"Coherence revision failed: {e}", level="error")

        # Fallback: return unchanged paper tex.
        return paper_tex

    def _extract_json(self, text: str) -> Any:
        t = text.strip()
        if t.startswith("{") and t.endswith("}"):
            return json.loads(t)
        m = re.search(r"\{[\s\S]+\}", t)
        if not m:
            raise ValueError("no JSON found")
        return json.loads(m.group(0))

