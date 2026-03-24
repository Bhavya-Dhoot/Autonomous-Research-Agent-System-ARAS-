from __future__ import annotations

import json
import re
from typing import Any

from aras.agents.base import BaseAgent, EventSink
from aras.config import Settings
from aras.healing.fallback_router import FallbackRouter
from aras.agents.memory_agent import MemoryAgent


class ResearchAgent(BaseAgent):
    """Research planning and hypothesis/experiment design agent."""

    def __init__(
        self,
        settings: Settings,
        memory: MemoryAgent,
        on_event: EventSink,
        on_tokens=None,
        on_chat_result=None,
    ) -> None:
        super().__init__(agent_id="research", on_event=on_event, on_tokens=on_tokens, on_chat_result=on_chat_result)
        self.settings = settings
        self.memory = memory
        self.router = FallbackRouter(settings=settings)

    async def plan(self, *, topic: str) -> dict[str, Any]:
        """Create a structured research plan for a topic."""
        prompts = self.memory.current_prompts()
        rag = self.memory.rag_context(query=f"research topic: {topic}")
        system = f"{prompts.get('research','')}\n\nRAG CONTEXT:\n{rag}\n\nReturn STRICT JSON."
        user = (
            "Create a research plan with:\n"
            "- hypothesis\n"
            "- key questions\n"
            "- datasets/benchmarks (or synthetic if none)\n"
            "- 3-5 experiments\n"
            "- evaluation metrics\n"
            "- target venues and IEEE/ACM section outline\n"
            "Return JSON with keys: hypothesis, questions, experiments, metrics, outline, keywords.\n"
            f"Topic: {topic}"
        )
        try:
            res = await self.router.chat(
                role_system=system,
                messages=[{"role": "user", "content": user}],
                purpose="planning",
                prefer=["nvidia", "openai", "local"],
                thinking=True,
                temperature=0.7,
                max_tokens=2500,
            )
            self.record_chat_result(res)
            self.add_tokens(res.tokens_used)
            self.emit(f"Plan generated via {res.provider}/{res.model}")
            obj = _extract_json(res.text)
            if not isinstance(obj, dict):
                raise ValueError("plan is not an object")
            return obj
        except Exception as e:
            self.emit(f"Plan fallback (no LLM): {e}", level="error")
            return {
                "hypothesis": f"A lightweight, reproducible baseline for '{topic}' can be validated with synthetic experiments and citation-backed analysis.",
                "questions": ["What are the dominant approaches?", "What metrics matter?", "What is reproducible quickly?"],
                "experiments": [
                    {"name": "baseline_benchmark", "goal": "Create a minimal baseline and measure runtime/accuracy proxy."},
                    {"name": "ablation_sensitivity", "goal": "Ablate key parameter(s) and observe changes."},
                    {"name": "robustness_noise", "goal": "Evaluate robustness under input noise/perturbation."},
                ],
                "metrics": ["time_seconds", "memory_mb", "score"],
                "outline": [
                    "Abstract",
                    "Introduction",
                    "Related Work",
                    "Methodology",
                    "Experiments",
                    "Results",
                    "Discussion",
                    "Conclusion",
                ],
                "keywords": [w for w in re.split(r"\\W+", topic.lower()) if w][:8],
            }


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    m = re.search(r"\{[\s\S]+\}", text)
    if not m:
        raise ValueError("no JSON found")
    return json.loads(m.group(0))

