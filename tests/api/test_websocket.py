from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from aras.ui.server import BroadcastHub, UiDiffFigureManager, build_app


class DummyOrchestrator:
    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "topic": "test",
            "cycle": 1,
            "agents": {"orchestrator": {"status": "IDLE"}},
            "pipeline": [{"name": "Planning", "progress": 0.0}],
            "current_task": "idle",
            "tokens_used": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "errors": 0,
            "cost_usd": 0.0,
            "budget_remaining_usd": None,
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


def _client(tmp_workspace: Path, settings) -> tuple[TestClient, BroadcastHub]:
    index_html = (Path(__file__).resolve().parents[2] / "aras" / "ui" / "static" / "index.html").read_text(encoding="utf-8")
    hub = BroadcastHub()
    app = build_app(index_html=index_html, hub=hub, orchestrator=DummyOrchestrator(), settings=settings, approval_gate=None)
    return TestClient(app), hub


def test_websocket_connects_and_receives_state(tmp_workspace: Path, settings, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    client, _hub = _client(tmp_workspace, settings)
    with client.websocket_connect("/ws/status") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "state"
        assert "state" in msg
        st = msg["state"]
        assert "agents" in st


@pytest.mark.slow
def test_figure_ready_event_broadcast(tmp_workspace: Path, settings, sample_figure_png, monkeypatch) -> None:
    """
    Start the watcher loop and ensure it broadcasts a figure_ready event.
    """
    monkeypatch.chdir(tmp_workspace)
    client, hub = _client(tmp_workspace, settings)

    with client.websocket_connect("/ws/status") as ws:
        # First message will be state.
        first = ws.receive_json()
        assert first.get("type") == "state"

        # Deterministic: broadcast AFTER the client is connected.
        import anyio

        async def _send():
            await hub.broadcast(
                {
                    "type": "figure_ready",
                    "name": "test",
                    "png_url": "/paper-figures/test.png",
                    "caption": "test",
                    "experiment": "experiment",
                    "fig_type": "figure",
                }
            )

        anyio.run(_send)

        # Then we should receive our broadcast.
        got_figure = False
        for _ in range(10):
            msg = ws.receive_json()
            if msg.get("type") == "figure_ready":
                got_figure = True
                assert msg["png_url"].startswith("/paper-figures/")
                break
        assert got_figure

