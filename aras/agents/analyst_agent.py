from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aras.agents.base import BaseAgent, EventSink
from aras.agents.memory_agent import MemoryAgent
from aras.config import Settings
from aras.healing.fallback_router import FallbackRouter
from aras.utils.fs import safe_write_text


class AnalystAgent(BaseAgent):
    """Analyze experiment results and produce structured conclusions."""

    def __init__(
        self,
        settings: Settings,
        memory: MemoryAgent,
        on_event: EventSink,
        on_tokens=None,
        on_chat_result=None,
    ) -> None:
        super().__init__(agent_id="analyst", on_event=on_event, on_tokens=on_tokens, on_chat_result=on_chat_result)
        self.settings = settings
        self.memory = memory
        self.router = FallbackRouter(settings=settings)

    async def analyze(
        self,
        *,
        topic: str,
        plan: dict[str, Any],
        scraped: list[dict[str, Any]],
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze results and generate tables/summary."""
        runs = results.get("runs") or []
        rows: list[dict[str, Any]] = []
        for r in runs:
            m = r.get("metrics") or {}
            rows.append(
                {
                    "experiment": r.get("name"),
                    "acc": m.get("acc"),
                    "final_loss": m.get("final_loss"),
                    "wall_s": r.get("wall_seconds"),
                    "peak_rss_mb": r.get("peak_rss_mb"),
                    "exit_code": r.get("exit_code"),
                }
            )
        df = pd.DataFrame(rows)
        table_md = df.to_markdown(index=False) if not df.empty else "(no results)"

        # Attempt LLM-based narrative analysis; fallback to heuristic summary.
        narrative = ""
        try:
            prompts = self.memory.current_prompts()
            rag = self.memory.rag_context(query=f"analysis for {topic}")
            system = f"{prompts.get('analyst','')}\n\nRAG CONTEXT:\n{rag}\n\nBe concrete."
            user = (
                "Analyze the following experiment table and summarize:\n"
                "- key takeaways\n"
                "- limitations\n"
                "- recommended next experiments\n\n"
                f"TOPIC: {topic}\n\nTABLE:\n{table_md}\n"
            )
            res = await self.router.chat(
                role_system=system,
                messages=[{"role": "user", "content": user}],
                purpose="analysis",
                prefer=["local", "nvidia", "openai"],
                thinking=False,
                temperature=0.4,
                max_tokens=900,
            )
            self.record_chat_result(res)
            self.add_tokens(res.tokens_used)
            narrative = res.text.strip()
            self.emit(f"Analysis generated via {res.provider}/{res.model}")
        except Exception as e:
            self.emit(f"Analysis fallback: {e}", level="error")
            narrative = _heuristic_narrative(df)

        out = {
            "table_markdown": table_md,
            "narrative": narrative,
            "lessons_learned": _lessons(df),
        }
        safe_write_text(Path("logs") / "analysis.json", json.dumps(out, ensure_ascii=False, indent=2))
        return out


def _heuristic_narrative(df: pd.DataFrame) -> str:
    if df.empty:
        return "No experiment results were produced; validate environment dependencies and rerun."
    best = df.sort_values("acc", ascending=False).iloc[0].to_dict()
    return (
        "Across synthetic experiments, accuracy is generally high, suggesting the baseline is stable. "
        f"Best run: {best.get('experiment')} with acc={best.get('acc')}. "
        "Runtime and memory are modest; further work should add real datasets and stronger baselines."
    )


def _lessons(df: pd.DataFrame) -> str:
    if df.empty:
        return "Ensure experiment runner is functioning; add environment checks and dependency pinning."
    return "Synthetic baselines are fast and reproducible; expand to real-world benchmarks to improve external validity."

