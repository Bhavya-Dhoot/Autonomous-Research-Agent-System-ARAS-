from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from aras.agents.novelty_agent import NoveltyAgent


class _FakeStore:
    def query(self, *, collection: str, text: str, n: int = 6) -> list[dict[str, Any]]:
        _ = (collection, text, n)
        return []


class _FakeMemory:
    def __init__(self) -> None:
        self.store = _FakeStore()

    def current_prompts(self) -> dict[str, str]:
        return {"novelty": "Return strict JSON only."}

    def rag_context(self, *, query: str, collections: list[str] | None = None) -> str:
        _ = (query, collections)
        return ""


class _DummyRouter:
    async def chat(self, **kwargs):
        _ = kwargs

        class _R:
            text = json.dumps(
                {
                    "original_topic": "network-novelty",
                    "selected_angle": "narrow to robust calibration under domain shift",
                    "novelty_score": 0.2,
                    "confidence": 0.9,
                    "competing_papers": [],
                    "pivot_reason": "dense prior work suggests tighter angle",
                }
            )

        return _R()


def _network_reachable() -> bool:
    urls = [
        "https://api.crossref.org/works?rows=1",
        "https://api.semanticscholar.org/graph/v1/paper/search?query=machine+learning&limit=1",
    ]
    for u in urls:
        try:
            with httpx.Client(timeout=8.0) as client:
                r = client.get(u, headers={"User-Agent": "ARAS/1.0"})
                if r.status_code < 500:
                    return True
        except Exception:
            continue
    return False


@pytest.mark.network
@pytest.mark.asyncio
async def test_novelty_evidence_ingestion_from_real_apis(tmp_workspace: Path, settings, monkeypatch: pytest.MonkeyPatch, sink_fn) -> None:
    if not _network_reachable():
        pytest.skip("No reachable novelty evidence endpoints")

    # Keep this opt-in and transparent for Crossref etiquette.
    crossref_email = os.environ.get("CROSSREF_EMAIL") or ""
    if not crossref_email:
        pytest.skip("CROSSREF_EMAIL not set")

    monkeypatch.chdir(tmp_workspace)
    settings.crossref_email = crossref_email

    agent = NoveltyAgent(settings=settings, memory=_FakeMemory(), on_event=sink_fn)  # type: ignore[arg-type]
    agent.router = _DummyRouter()  # type: ignore[assignment]

    result = await agent.check(
        topic="calibration of confidence in text classification",
        plan={
            "hypothesis": "temperature scaling under shift improves reliability",
            "keywords": ["calibration", "uncertainty", "confidence", "text classification"],
        },
        cycle=99,
    )

    # Core functional assertions: real evidence ingestion happened.
    assert result.evidence_count > 0
    assert len(result.competing_papers) > 0
    assert any(src in {"crossref", "semantic_scholar"} for src in result.evidence_sources)

    # Ensure normalized schema shape in at least one competing paper.
    sample = result.competing_papers[0]
    assert isinstance(sample.get("title"), str) and sample.get("title")
    assert "url" in sample

    # Audit artifact should exist for UI/debug traceability.
    ev_path = tmp_workspace / "logs" / "novelty_evidence_cycle99.json"
    assert ev_path.exists()
    payload = json.loads(ev_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and payload
