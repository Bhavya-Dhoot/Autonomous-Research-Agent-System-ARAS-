from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from aras.ui.server import BroadcastHub, build_app


class DummyOrchestrator:
    def ui_state(self):
        return {
            "agents": {},
            "pipeline": [],
            "current_task": "-",
            "tokens_used": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "errors": 0,
            "cost_usd": 0.0,
            "budget_remaining_usd": None,
            "cost_per_cycle": 0.0,
            "topic": "-",
            "cycle": 0,
            "paper_preview": "",
            "memory_preview": "",
            "github_url": None,
            "hf_url": None,
            "paper_score": None,
        }

    async def ui_logs(self):
        if False:
            yield {}


def _client(tmp_workspace: Path, settings) -> TestClient:
    index_html = (Path(__file__).resolve().parents[2] / "aras" / "ui" / "static" / "index.html").read_text(encoding="utf-8")
    app = build_app(index_html=index_html, hub=BroadcastHub(), orchestrator=DummyOrchestrator(), settings=settings, approval_gate=None)
    return TestClient(app)


def test_quality_endpoint_empty(tmp_workspace: Path, settings, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    r = _client(tmp_workspace, settings).get("/api/quality")
    assert r.status_code == 200
    obj = r.json()
    assert obj["latest"] is None
    assert obj["history_count"] == 0


def test_quality_endpoint_latest_row(tmp_workspace: Path, settings, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    logs = tmp_workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    p = logs / "cycle_quality.jsonl"
    rows = [
        {"cycle": 1, "improvement_index": 4.0, "novelty": {"gate_passed": False}},
        {"cycle": 2, "improvement_index": 6.2, "novelty": {"gate_passed": True}},
    ]
    p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    r = _client(tmp_workspace, settings).get("/api/quality")
    assert r.status_code == 200
    obj = r.json()
    assert obj["history_count"] == 2
    assert obj["latest"]["cycle"] == 2
    assert obj["latest"]["improvement_index"] == 6.2
