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


def test_discovers_nested_pngs(tmp_workspace: Path, settings, sample_figure_png, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    nested = tmp_workspace / "paper" / "figures" / "subdir"
    nested.mkdir(parents=True, exist_ok=True)
    target = nested / "nested.png"
    target.write_bytes(sample_figure_png.read_bytes())
    r = _client(tmp_workspace, settings).get("/api/figures")
    assert r.status_code == 200
    names = {x["name"] for x in r.json()}
    assert "nested" in names


def test_reads_captions_json(tmp_workspace: Path, settings, sample_figure_png, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    caps = tmp_workspace / "paper" / "figures" / "captions.json"
    caps.write_text('{"test":"hello caption"}', encoding="utf-8")
    r = _client(tmp_workspace, settings).get("/api/figures")
    item = r.json()[0]
    assert item["caption"] == "hello caption"


def test_reads_structured_captions_json(tmp_workspace: Path, settings, sample_figure_png, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    caps = tmp_workspace / "paper" / "figures" / "captions.json"
    caps.write_text('{"test":{"caption":"structured caption","confidence":"high"}}', encoding="utf-8")
    r = _client(tmp_workspace, settings).get("/api/figures")
    item = r.json()[0]
    assert item["caption"] == "structured caption"


def test_only_pngs_returned(tmp_workspace: Path, settings, sample_figure_png, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    figs = tmp_workspace / "paper" / "figures"
    (figs / "x.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
    (figs / "y.eps").write_text("%!PS-Adobe-3.0\n", encoding="utf-8")
    r = _client(tmp_workspace, settings).get("/api/figures")
    assert r.status_code == 200
    assert all(x["png_url"].endswith(".png") for x in r.json())


def test_static_serving_correct_content(tmp_workspace: Path, settings, sample_figure_png, monkeypatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    r = _client(tmp_workspace, settings).get("/paper-figures/test.png")
    assert r.status_code == 200
    assert r.content == sample_figure_png.read_bytes()
