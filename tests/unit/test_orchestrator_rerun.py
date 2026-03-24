from __future__ import annotations

from pathlib import Path

import pytest

from aras.orchestrator import Orchestrator


@pytest.mark.asyncio
async def test_cycle_reruns_when_all_runs_degraded(tmp_workspace: Path, settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    settings.figure_quality_rerun_enabled = True
    settings.figure_quality_max_reruns = 2

    o = Orchestrator(settings=settings)

    calls = {"runs": 0}

    async def _plan(*, topic: str):
        return {"hypothesis": "h", "keywords": []}

    async def _novelty(**kwargs):
        class N:
            novelty_score = 0.9
            selected_angle = None

        return N()

    async def _scrape(*, plan):
        return []

    async def _validate(*, items, cycle):
        return items

    async def _design(**kwargs):
        return object()

    async def _run_experiments(**kwargs):
        calls["runs"] += 1
        return {"runs": [{"name": f"exp_{calls['runs']}", "exit_code": 0, "metrics": {"losses": [1.0], "acc": 0.2}}]}

    async def _analyze(**kwargs):
        return {"table_markdown": "", "narrative": "", "lessons_learned": ""}

    async def _figures(**kwargs):
        all_bad = calls["runs"] < 2
        return {
            "figures_latex": "",
            "architecture_tikz": "",
            "figures_paths": {},
            "figure_quality_summary": {
                "high_confidence": 0 if all_bad else 1,
                "low_confidence": 1 if all_bad else 0,
                "excluded_from_paper": 1 if all_bad else 0,
                "all_runs_degraded": all_bad,
                "health": {"success": 0 if all_bad else 1, "degraded": 1 if all_bad else 0, "failed": 0},
            },
        }

    async def _write(**kwargs):
        paper = tmp_workspace / "paper" / "paper.tex"
        paper.write_text("\\begin{document}x\\end{document}", encoding="utf-8")
        return {"tex": paper, "pdf": None}

    async def _review(**kwargs):
        return {"score": 0.0}

    async def _revise(**kwargs):
        return "\\begin{document}x\\end{document}"

    async def _score(**kwargs):
        return 0.1

    async def _store_cycle(**kwargs):
        return None

    async def _preview():
        return "preview"

    async def _snapshot(**kwargs):
        return None

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
    monkeypatch.setattr(o.memory, "store_cycle", _store_cycle)
    monkeypatch.setattr(o.memory, "preview", _preview)
    monkeypatch.setattr(o.memory, "snapshot", _snapshot)
    monkeypatch.setattr(o.prompt_evolver, "evolve", _evolve)
    monkeypatch.setattr(o.ab_tester, "test_writer_abstract", _ab)
    monkeypatch.setattr(o.github, "publish", _publish)
    monkeypatch.setattr(o.memory, "store_citations", _snapshot)

    await o._run_cycle(topic="t", cycle=1, outputs={})
    assert calls["runs"] == 2


@pytest.mark.asyncio
async def test_cycle_stops_rerun_when_disabled(tmp_workspace: Path, settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    settings.figure_quality_rerun_enabled = False
    settings.figure_quality_max_reruns = 2

    o = Orchestrator(settings=settings)

    calls = {"runs": 0}

    async def _plan(*, topic: str):
        return {"hypothesis": "h", "keywords": []}

    async def _novelty(**kwargs):
        class N:
            novelty_score = 0.9
            selected_angle = None

        return N()

    async def _scrape(*, plan):
        return []

    async def _validate(*, items, cycle):
        return items

    async def _design(**kwargs):
        return object()

    async def _run_experiments(**kwargs):
        calls["runs"] += 1
        return {"runs": [{"name": "exp", "exit_code": 0, "metrics": {"losses": [1.0], "acc": 0.2}}]}

    async def _analyze(**kwargs):
        return {"table_markdown": "", "narrative": "", "lessons_learned": ""}

    async def _figures(**kwargs):
        return {
            "figures_latex": "",
            "architecture_tikz": "",
            "figures_paths": {},
            "figure_quality_summary": {
                "high_confidence": 0,
                "low_confidence": 1,
                "excluded_from_paper": 1,
                "all_runs_degraded": True,
                "health": {"success": 0, "degraded": 1, "failed": 0},
            },
        }

    async def _write(**kwargs):
        paper = tmp_workspace / "paper" / "paper.tex"
        paper.write_text("\\begin{document}x\\end{document}", encoding="utf-8")
        return {"tex": paper, "pdf": None}

    async def _review(**kwargs):
        return {"score": 0.0}

    async def _revise(**kwargs):
        return "\\begin{document}x\\end{document}"

    async def _score(**kwargs):
        return 0.1

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

    await o._run_cycle(topic="t", cycle=1, outputs={})
    assert calls["runs"] == 1
