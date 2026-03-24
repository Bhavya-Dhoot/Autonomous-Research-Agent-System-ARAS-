from __future__ import annotations

import pytest

from aras.agents.novelty_agent import NoveltyAgent
from aras.types import NoveltyResult


@pytest.mark.asyncio
async def test_strict_gate_blocks_without_validated_evidence(settings, sink_fn) -> None:
    agent = NoveltyAgent(settings=settings, memory=None, on_event=sink_fn)  # type: ignore[arg-type]

    # Stub evidence + merge path to avoid network.
    async def _gather(**kwargs):
        return [{"title": "A", "year": "2023", "url": "u", "source": "semantic_scholar", "validated": False, "citation_count": 0}]

    def _merge(**kwargs):
        return NoveltyResult(
            original_topic="t",
            selected_angle="new-angle",
            novelty_score=0.2,
            competing_papers=[{"title": "A", "year": "2023", "url": "u"}],
            pivot_reason="x",
            confidence=0.9,
            evidence_count=1,
            validated_evidence_count=0,
            evidence_sources=["semantic_scholar"],
            gate_passed=False,
            gate_reason=None,
        )

    class _Router:
        async def chat(self, **kwargs):
            raise RuntimeError("skip llm")

    agent.router = _Router()  # type: ignore[assignment]
    agent._gather_evidence = _gather  # type: ignore[method-assign]
    agent._merge_novelty = _merge  # type: ignore[method-assign]

    out = await agent.check(topic="t", plan={"hypothesis": "h", "keywords": ["k"]}, cycle=1)
    assert out.gate_passed is False
    assert out.gate_reason == "insufficient_validated_evidence"
    assert out.selected_angle == ""


@pytest.mark.asyncio
async def test_strict_gate_passes_with_thresholds(settings, sink_fn) -> None:
    agent = NoveltyAgent(settings=settings, memory=None, on_event=sink_fn)  # type: ignore[arg-type]

    async def _gather(**kwargs):
        return []

    def _merge(**kwargs):
        return NoveltyResult(
            original_topic="t",
            selected_angle="better-angle",
            novelty_score=0.2,
            competing_papers=[{"title": "A", "year": "2023", "url": "u"}],
            pivot_reason="x",
            confidence=0.8,
            evidence_count=5,
            validated_evidence_count=3,
            evidence_sources=["crossref", "semantic_scholar"],
            gate_passed=False,
            gate_reason=None,
        )

    class _Router:
        async def chat(self, **kwargs):
            raise RuntimeError("skip llm")

    agent.router = _Router()  # type: ignore[assignment]
    agent._gather_evidence = _gather  # type: ignore[method-assign]
    agent._merge_novelty = _merge  # type: ignore[method-assign]

    out = await agent.check(topic="t", plan={"hypothesis": "h", "keywords": ["k"]}, cycle=1)
    assert out.gate_passed is True
    assert out.gate_reason == "strict_evidence_gate_passed"
    assert out.selected_angle == "better-angle"
