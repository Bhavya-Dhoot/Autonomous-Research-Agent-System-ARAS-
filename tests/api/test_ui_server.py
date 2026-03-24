from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from aras.ui.server import BroadcastHub, build_app


class DummyOrchestrator:
    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "topic": "test",
            "cycle": 1,
            "agents": {},
            "pipeline": [],
            "current_task": "idle",
            "tokens_used": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "errors": 0,
            "cost_usd": 0.0,
            "budget_remaining_usd": 1.0,
            "cost_per_cycle": 0.0,
            "paper_preview": "",
            "memory_preview": "",
            "github_url": None,
            "hf_url": None,
            "paper_score": None,
        }

    def ui_state(self) -> dict[str, Any]:
        return dict(self._state)

    async def ui_logs(self):
        if False:
            yield {}


def _make_client(tmp_workspace: Path, settings) -> TestClient:
    index_html = (Path(__file__).resolve().parents[2] / "aras" / "ui" / "static" / "index.html").read_text(encoding="utf-8")
    hub = BroadcastHub()
    app = build_app(index_html=index_html, hub=hub, orchestrator=DummyOrchestrator(), settings=settings, approval_gate=None)
    return TestClient(app)


def test_get_root_returns_html(tmp_workspace: Path, settings) -> None:
    client = _make_client(tmp_workspace, settings)
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "<html" in r.text.lower()


def test_get_root_contains_tabs(tmp_workspace: Path, settings) -> None:
    client = _make_client(tmp_workspace, settings)
    r = client.get("/")
    assert r.status_code == 200
    assert "DIFFS" in r.text
    assert "FIGURES" in r.text


def test_diffs_endpoint_empty_dir(tmp_workspace: Path, settings, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    client = _make_client(tmp_workspace, settings)
    r = client.get("/api/diffs")
    assert r.status_code == 200
    assert r.json() == []


def test_diffs_endpoint_with_patch(tmp_workspace: Path, settings, sample_diff_patch, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    client = _make_client(tmp_workspace, settings)
    r = client.get("/api/diffs")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) == 1
    item = data[0]
    assert item["round"] == 1
    assert item["lines_added"] >= 0
    assert item["lines_removed"] >= 0
    assert isinstance(item.get("summary"), str) and item["summary"]


def test_figures_endpoint_empty_dir(tmp_workspace: Path, settings, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    client = _make_client(tmp_workspace, settings)
    r = client.get("/api/figures")
    assert r.status_code == 200
    assert r.json() == []


def test_figures_endpoint_with_png(tmp_workspace: Path, settings, sample_figure_png, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    client = _make_client(tmp_workspace, settings)
    r = client.get("/api/figures")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) == 1
    item = data[0]
    assert item["name"] == "test"
    assert item["png_url"].startswith("/paper-figures/")
    assert "experiment" in item


def test_paper_figures_static_serve(tmp_workspace: Path, settings, sample_figure_png, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    client = _make_client(tmp_workspace, settings)
    r = client.get("/paper-figures/test.png")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("image/")
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

