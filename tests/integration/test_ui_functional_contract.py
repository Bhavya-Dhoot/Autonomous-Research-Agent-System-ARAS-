from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from aras.ui.server import BroadcastHub, build_app


class DummyOrchestrator:
    def ui_state(self):
        return {
            "topic": "t",
            "cycle": 1,
            "agents": {"orchestrator": {"status": "WORKING", "detail": "x", "last_update": "now"}},
            "pipeline": [{"name": "Novelty", "progress": 0.5}],
            "current_task": "running",
            "tokens_used": 10,
            "tokens_input": 6,
            "tokens_output": 4,
            "errors": 0,
            "cost_usd": 0.1,
            "budget_remaining_usd": 9.9,
            "cost_per_cycle": 0.1,
            "paper_preview": "",
            "memory_preview": "",
            "github_url": None,
            "hf_url": None,
            "paper_score": 6.0,
        }

    async def ui_logs(self):
        if False:
            yield {}


def _client(tmp_workspace: Path, settings) -> TestClient:
    index_html = (Path(__file__).resolve().parents[2] / "aras" / "ui" / "static" / "index.html").read_text(encoding="utf-8")
    app = build_app(index_html=index_html, hub=BroadcastHub(), orchestrator=DummyOrchestrator(), settings=settings, approval_gate=None)
    return TestClient(app)


def test_api_quality_reflects_file(tmp_workspace: Path, settings, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    logs = tmp_workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    q = {
        "cycle": 1,
        "novelty": {"gate_passed": True, "gate_reason": "strict_evidence_gate_passed"},
        "improvement_index": 7.1,
    }
    (logs / "cycle_quality.jsonl").write_text(json.dumps(q) + "\n", encoding="utf-8")
    r = _client(tmp_workspace, settings).get("/api/quality")
    assert r.status_code == 200
    obj = r.json()
    assert obj["latest"]["cycle"] == 1
    assert obj["latest"]["novelty"]["gate_passed"] is True


def test_api_figures_reflects_real_files_only(tmp_workspace: Path, settings, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    figs = tmp_workspace / "paper" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    (figs / "fake.txt").write_text("not a figure", encoding="utf-8")
    # Valid PNG signature only.
    (figs / "real.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    r = _client(tmp_workspace, settings).get("/api/figures")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["name"] == "real"
