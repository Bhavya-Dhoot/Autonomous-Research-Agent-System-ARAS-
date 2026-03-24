from __future__ import annotations

import json
from pathlib import Path

import pytest

from aras.orchestrator import Orchestrator
from aras.types import NoveltyResult


@pytest.mark.asyncio
async def test_no_pivot_when_strict_gate_fails(tmp_workspace: Path, settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    settings.review_rounds = 1

    o = Orchestrator(settings=settings)

    seen_topics: list[str] = []

    async def _plan(*, topic: str):
        seen_topics.append(topic)
        return {"hypothesis": "h", "keywords": ["k"]}

    async def _novelty(**kwargs):
        return NoveltyResult(
            original_topic="t",
            selected_angle="should-not-apply",
            novelty_score=0.2,
            competing_papers=[{"title": "A", "year": "2023", "url": "u"}],
            pivot_reason="x",
            confidence=0.95,
            evidence_count=2,
            validated_evidence_count=1,
            evidence_sources=["crossref"],
            gate_passed=False,
            gate_reason="insufficient_validated_evidence",
        )

    async def _scrape(*, plan):
        return []

    async def _validate(*, items, cycle):
        return []

    async def _design(**kwargs):
        return object()

    async def _run_experiments(**kwargs):
        return {"runs": [{"name": "exp", "exit_code": 0, "metrics": {"losses": [1.0, 0.9], "acc": 0.8}}]}

    async def _analyze(**kwargs):
        return {"table_markdown": "", "narrative": "", "lessons_learned": ""}

    async def _figures(**kwargs):
        return {
            "figures_latex": "",
            "architecture_tikz": "",
            "figures_paths": {},
            "figure_quality_summary": {
                "high_confidence": 1,
                "low_confidence": 0,
                "excluded_from_paper": 0,
                "all_runs_degraded": False,
                "health": {"success": 1, "degraded": 0, "failed": 0},
            },
        }

    async def _write(**kwargs):
        paper = tmp_workspace / "paper" / "paper.tex"
        paper.write_text("\\begin{abstract}x\\end{abstract}\\begin{document}x\\end{document}", encoding="utf-8")
        return {"tex": paper, "pdf": None}

    async def _review(**kwargs):
        return {"score": 5.5, "overall_score": 5.5}

    async def _revise(**kwargs):
        return "\\begin{abstract}x\\end{abstract}\\begin{document}x\\end{document}"

    async def _score(**kwargs):
        return 5.5

    async def _noop(**kwargs):
        return None

    async def _preview():
        return "preview"

    async def _evolve(**kwargs):
        return {"lessons_learned": "ok", "prompt_version": "v1"}

    async def _ab(**kwargs):
        return {"winner": "a", "score_a": 1.0, "score_b": 0.9}

    async def _publish(**kwargs):
        return {"url": None}

    monkeypatch.setattr(o.research, "plan", _plan)
    monkeypatch.setattr(o.novelty, "check", _novelty)
    monkeypatch.setattr(o.scraping, "scrape", _scrape)
    monkeypatch.setattr(o.citations, "validate", _validate)
    monkeypatch.setattr(o.coder, "design_and_write_experiments", _design)
    monkeypatch.setattr(o.coder, "run_experiments", _run_experiments)
    monkeypatch.setattr(o.analyst, "analyze", _analyze)
    monkeypatch.setattr(o.figures, "generate", _figures)
    monkeypatch.setattr(o.writer, "write_paper", _write)
    monkeypatch.setattr(o.reviewer, "review", _review)
    monkeypatch.setattr(o.coherence, "revise", _revise)
    monkeypatch.setattr(o.scorer, "score", _score)
    monkeypatch.setattr(o.memory, "store_cycle", _noop)
    monkeypatch.setattr(o.memory, "preview", _preview)
    monkeypatch.setattr(o.memory, "snapshot", _noop)
    monkeypatch.setattr(o.prompt_evolver, "evolve", _evolve)
    monkeypatch.setattr(o.ab_tester, "test_writer_abstract", _ab)
    monkeypatch.setattr(o.github, "publish", _publish)
    monkeypatch.setattr(o.memory, "store_citations", _noop)

    await o._run_cycle(topic="base-topic", cycle=1, outputs={})
    # Only initial plan should be called; no pivot re-plan.
    assert seen_topics == ["base-topic"]


@pytest.mark.asyncio
async def test_cycle_quality_written_and_ui_quality_endpoint_reads(tmp_workspace: Path, settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    settings.review_rounds = 1

    o = Orchestrator(settings=settings)

    async def _plan(*, topic: str):
        return {"hypothesis": "h", "keywords": ["k"]}

    async def _novelty(**kwargs):
        return NoveltyResult(
            original_topic="t",
            selected_angle="",
            novelty_score=0.6,
            competing_papers=[],
            pivot_reason=None,
            confidence=0.8,
            evidence_count=4,
            validated_evidence_count=3,
            evidence_sources=["crossref", "semantic_scholar"],
            gate_passed=False,
            gate_reason="novelty_not_low_enough",
        )

    async def _scrape(*, plan):
        return []

    async def _validate(*, items, cycle):
        return []

    async def _design(**kwargs):
        return object()

    async def _run_experiments(**kwargs):
        return {"runs": [{"name": "exp", "exit_code": 0, "metrics": {"losses": [1.0, 0.7], "acc": 0.9}}]}

    async def _analyze(**kwargs):
        return {"table_markdown": "", "narrative": "", "lessons_learned": ""}

    async def _figures(**kwargs):
        return {
            "figures_latex": "",
            "architecture_tikz": "",
            "figures_paths": {},
            "figure_quality_summary": {
                "high_confidence": 2,
                "low_confidence": 1,
                "excluded_from_paper": 1,
                "all_runs_degraded": False,
                "health": {"success": 1, "degraded": 0, "failed": 0},
            },
        }

    async def _write(**kwargs):
        paper = tmp_workspace / "paper" / "paper.tex"
        paper.write_text("\\begin{abstract}x\\end{abstract}\\begin{document}x\\end{document}", encoding="utf-8")
        return {"tex": paper, "pdf": None}

    async def _review(**kwargs):
        return {"score": 6.0, "overall_score": 6.0}

    async def _revise(**kwargs):
        return "\\begin{abstract}x\\end{abstract}\\begin{document}x\\end{document}"

    async def _score(**kwargs):
        return 6.0

    async def _noop(**kwargs):
        return None

    async def _preview():
        return "preview"

    async def _evolve(**kwargs):
        return {"lessons_learned": "ok", "prompt_version": "v1"}

    async def _ab(**kwargs):
        return {"winner": "a", "score_a": 1.0, "score_b": 0.9}

    async def _publish(**kwargs):
        return {"url": None}

    monkeypatch.setattr(o.research, "plan", _plan)
    monkeypatch.setattr(o.novelty, "check", _novelty)
    monkeypatch.setattr(o.scraping, "scrape", _scrape)
    monkeypatch.setattr(o.citations, "validate", _validate)
    monkeypatch.setattr(o.coder, "design_and_write_experiments", _design)
    monkeypatch.setattr(o.coder, "run_experiments", _run_experiments)
    monkeypatch.setattr(o.analyst, "analyze", _analyze)
    monkeypatch.setattr(o.figures, "generate", _figures)
    monkeypatch.setattr(o.writer, "write_paper", _write)
    monkeypatch.setattr(o.reviewer, "review", _review)
    monkeypatch.setattr(o.coherence, "revise", _revise)
    monkeypatch.setattr(o.scorer, "score", _score)
    monkeypatch.setattr(o.memory, "store_cycle", _noop)
    monkeypatch.setattr(o.memory, "preview", _preview)
    monkeypatch.setattr(o.memory, "snapshot", _noop)
    monkeypatch.setattr(o.prompt_evolver, "evolve", _evolve)
    monkeypatch.setattr(o.ab_tester, "test_writer_abstract", _ab)
    monkeypatch.setattr(o.github, "publish", _publish)
    monkeypatch.setattr(o.memory, "store_citations", _noop)

    await o._run_cycle(topic="quality-topic", cycle=1, outputs={})

    p = tmp_workspace / "logs" / "cycle_quality.jsonl"
    assert p.exists()
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines
    row = json.loads(lines[-1])
    assert row["cycle"] == 1
    assert row["novelty"]["gate_passed"] is False
    assert isinstance(row["improvement_index"], float)
