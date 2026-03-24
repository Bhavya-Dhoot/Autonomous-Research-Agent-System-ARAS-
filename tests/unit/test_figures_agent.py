from __future__ import annotations

import json
from pathlib import Path

import pytest

from aras.agents.figures_agent import FiguresAgent


@pytest.mark.asyncio
async def test_generate_creates_png_and_pdf(tmp_workspace: Path, settings, sample_results_json) -> None:
    out_dir = tmp_workspace / "paper"
    agent = FiguresAgent(settings=settings, on_event=lambda agent, message, level="info": None)
    res = await agent.generate(topic="Test Topic", results=sample_results_json, output_dir=out_dir)
    figs = out_dir / "figures"
    assert figs.exists()
    assert any(p.suffix.lower() == ".png" for p in figs.glob("*.png"))
    assert any(p.suffix.lower() == ".pdf" for p in figs.glob("*.pdf"))
    assert "architecture_tikz" in res
    assert "figure_quality_summary" in res
    assert isinstance(res.get("paper_eligible_figures"), list)


@pytest.mark.asyncio
async def test_generate_creates_diagnostic_when_no_valid_curves(tmp_workspace: Path, settings) -> None:
    out_dir = tmp_workspace / "paper"
    agent = FiguresAgent(settings=settings, on_event=lambda agent, message, level="info": None)
    results = {
        "runs": [
            {
                "name": "exp_scalar",
                "exit_code": 0,
                "metrics": {"acc": 0.8, "losses": [1.0]},
            }
        ]
    }
    res = await agent.generate(topic="Diag Topic", results=results, output_dir=out_dir)
    assert (out_dir / "figures" / "training_data_quality.png").exists()
    assert not (out_dir / "figures" / "loss_curves.png").exists()
    # Low-confidence diagnostics should be excluded from paper latex.
    assert "training_data_quality" not in " ".join([str(x) for x in res.get("paper_eligible_figures", [])])
    assert "fig:training_data_quality" not in str(res.get("figures_latex") or "")


@pytest.mark.asyncio
async def test_generate_copies_run_artifact_and_writes_captions(tmp_workspace: Path, settings, sample_figure_png: Path) -> None:
    out_dir = tmp_workspace / "paper"
    agent = FiguresAgent(settings=settings, on_event=lambda agent, message, level="info": None)
    source_png = tmp_workspace / "source_loss.png"
    source_png.write_bytes(sample_figure_png.read_bytes())
    results = {
        "runs": [
            {
                "name": "exp_copy_me",
                "exit_code": 0,
                "artifacts": {"loss.png": str(source_png.resolve())},
                "metrics": {"acc": 0.7, "losses": [1.0, 0.8, 0.6]},
            }
        ]
    }
    await agent.generate(topic="Artifact Topic", results=results, output_dir=out_dir)

    copied = out_dir / "figures" / "experiments" / "exp_copy_me.png"
    assert copied.exists()

    caps = out_dir / "figures" / "captions.json"
    assert caps.exists()
    obj = json.loads(caps.read_text(encoding="utf-8"))
    assert "exp_copy_me" in obj
    assert obj["exp_copy_me"]["confidence"] == "high"


@pytest.mark.asyncio
async def test_generate_marks_all_runs_degraded(tmp_workspace: Path, settings) -> None:
    out_dir = tmp_workspace / "paper"
    agent = FiguresAgent(settings=settings, on_event=lambda agent, message, level="info": None)
    results = {
        "runs": [
            {"name": "exp_a", "exit_code": 0, "metrics": {"losses": [1.0], "acc": 0.4}},
            {"name": "exp_b", "exit_code": 0, "metrics": {"losses": [0.0], "acc": 0.5}},
        ]
    }
    res = await agent.generate(topic="Degraded Topic", results=results, output_dir=out_dir)
    q = res.get("figure_quality_summary") or {}
    assert q.get("all_runs_degraded") is True
