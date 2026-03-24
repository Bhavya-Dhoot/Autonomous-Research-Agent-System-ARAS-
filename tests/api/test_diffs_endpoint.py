from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aras.ui.server import BroadcastHub, build_app


class DummyOrchestrator:
    def ui_state(self):
        return {"agents": {}, "pipeline": [], "current_task": "-", "tokens_used": 0, "tokens_input": 0, "tokens_output": 0, "errors": 0, "cost_usd": 0.0, "budget_remaining_usd": None, "cost_per_cycle": 0.0, "topic": "-", "cycle": 0, "paper_preview": "", "memory_preview": "", "github_url": None, "hf_url": None, "paper_score": None}

    async def ui_logs(self):
        if False:
            yield {}


def _client(tmp_workspace: Path, settings) -> TestClient:
    index_html = (Path(__file__).resolve().parents[2] / "aras" / "ui" / "static" / "index.html").read_text(encoding="utf-8")
    app = build_app(index_html=index_html, hub=BroadcastHub(), orchestrator=DummyOrchestrator(), settings=settings, approval_gate=None)
    return TestClient(app)


def test_parses_round_from_filename(tmp_workspace: Path, settings, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    diffs = tmp_workspace / "paper" / "diffs"
    diffs.mkdir(parents=True, exist_ok=True)
    (diffs / "paper_diff_round2.patch").write_text("--- a\n+++ b\n+X\n", encoding="utf-8")
    r = _client(tmp_workspace, settings).get("/api/diffs")
    assert r.status_code == 200
    data = r.json()
    assert data and data[0]["round"] == 2


def test_multiple_rounds_returned_sorted(tmp_workspace: Path, settings, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    diffs = tmp_workspace / "paper" / "diffs"
    diffs.mkdir(parents=True, exist_ok=True)
    (diffs / "paper_diff_round3.patch").write_text("--- a\n+++ b\n+X\n", encoding="utf-8")
    (diffs / "paper_diff_round1.patch").write_text("--- a\n+++ b\n+Y\n", encoding="utf-8")
    (diffs / "paper_diff_round2.patch").write_text("--- a\n+++ b\n+Z\n", encoding="utf-8")
    r = _client(tmp_workspace, settings).get("/api/diffs")
    rounds = [x["round"] for x in r.json()]
    assert rounds == sorted(rounds)


def test_non_patch_files_ignored(tmp_workspace: Path, settings, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    diffs = tmp_workspace / "paper" / "diffs"
    diffs.mkdir(parents=True, exist_ok=True)
    (diffs / "README.txt").write_text("ignore", encoding="utf-8")
    r = _client(tmp_workspace, settings).get("/api/diffs")
    assert r.status_code == 200
    assert r.json() == []


def test_empty_patch_file_handled(tmp_workspace: Path, settings, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    diffs = tmp_workspace / "paper" / "diffs"
    diffs.mkdir(parents=True, exist_ok=True)
    (diffs / "paper_diff_round1.patch").write_text("", encoding="utf-8")
    r = _client(tmp_workspace, settings).get("/api/diffs")
    assert r.status_code == 200
    item = r.json()[0]
    assert item["lines_added"] == 0
    assert item["lines_removed"] == 0

