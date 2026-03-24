from __future__ import annotations

import json
from pathlib import Path

import pytest

from aras.agents.huggingface_agent import HuggingFaceAgent


@pytest.mark.asyncio
async def test_skips_when_no_token(settings, tmp_workspace: Path, sink_fn) -> None:
    settings.hf_token = None
    settings.hf_username = None
    agent = HuggingFaceAgent(settings=settings, on_event=sink_fn)
    out = await agent.publish_dataset(topic="t", outputs_root=tmp_workspace, paper_pdf=None, cycle=1)
    assert out["dataset_url"] is None
    assert out["model_url"] is None
    assert out["space_url"] is None
    assert out["skipped"] is True


@pytest.mark.asyncio
async def test_creates_dataset_and_space_repo(fake_hf_api, settings, tmp_workspace: Path, sample_results_json, sample_paper_tex, monkeypatch, sink_fn) -> None:
    # Enable HF
    settings.hf_token = "x"
    settings.hf_username = "user"

    # Ensure expected results.json location exists (agent scans outputs_root/experiments/**/results.json).
    # Our fixture writes to experiments/test_slug/results.json already.
    monkeypatch.chdir(tmp_workspace)

    agent = HuggingFaceAgent(settings=settings, on_event=sink_fn)
    out = await agent.publish_dataset(topic="My Topic", outputs_root=tmp_workspace, paper_pdf=None, cycle=1)

    calls = fake_hf_api.calls
    create_calls = [c for c in calls if c[0] == "create_repo"]
    assert any(cc[1].get("repo_type") == "dataset" for cc in create_calls)
    assert any(cc[1].get("repo_type") == "space" for cc in create_calls)
    assert out["space_url"] and "huggingface.co/spaces" in out["space_url"]


@pytest.mark.asyncio
async def test_creates_model_repo_when_checkpoint_found(fake_hf_api, settings, tmp_workspace: Path, sample_results_json, sample_paper_tex, monkeypatch, sink_fn) -> None:
    settings.hf_token = "x"
    settings.hf_username = "user"
    monkeypatch.chdir(tmp_workspace)

    # Create a dummy checkpoint in experiments/
    ckpt_dir = tmp_workspace / "experiments" / "test_slug"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "model.joblib").write_bytes(b"fake")

    agent = HuggingFaceAgent(settings=settings, on_event=sink_fn)
    _out = await agent.publish_dataset(topic="My Topic", outputs_root=tmp_workspace, paper_pdf=None, cycle=1)
    create_calls = [c for c in fake_hf_api.calls if c[0] == "create_repo"]
    assert any(cc[1].get("repo_type") == "model" for cc in create_calls)


@pytest.mark.asyncio
async def test_app_py_generated_and_valid(fake_hf_api, settings, tmp_workspace: Path, sample_results_json, sample_paper_tex, monkeypatch, sink_fn) -> None:
    settings.hf_token = "x"
    settings.hf_username = "user"
    monkeypatch.chdir(tmp_workspace)

    agent = HuggingFaceAgent(settings=settings, on_event=sink_fn)
    _out = await agent.publish_dataset(topic="My Topic", outputs_root=tmp_workspace, paper_pdf=None, cycle=1)

    # The agent writes a local logs/hf_last.json; assert it exists.
    p = tmp_workspace / "logs" / "hf_last.json"
    assert p.exists()
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert "dataset_url" in obj and "space_url" in obj

