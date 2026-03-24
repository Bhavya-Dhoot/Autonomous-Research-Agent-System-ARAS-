from __future__ import annotations

import asyncio
import contextlib
import json
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from aras.config import Settings
from aras.healing.fallback_router import ChatResult, FallbackRouter


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Path:
    """
    Create a full ARAS workspace directory layout under tmp_path.
    """
    (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)
    (tmp_path / "paper" / "figures").mkdir(parents=True, exist_ok=True)
    (tmp_path / "paper" / "diffs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory_snapshot").mkdir(parents=True, exist_ok=True)
    (tmp_path / "prompt_versions").mkdir(parents=True, exist_ok=True)
    (tmp_path / "chroma_db").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def settings(tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """
    Settings configured for fast, offline-by-default tests.
    """
    # Ensure Settings loads from the temp workspace .env if present (we generally won't create one).
    monkeypatch.chdir(tmp_workspace)

    # Provide env overrides for aliases used by pydantic-settings.
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_workspace / "chroma_db"))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:16379")  # invalid -> forces no-cache paths
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HF_USERNAME", "")
    monkeypatch.setenv("BUDGET_USD_CEILING", "10.0")
    monkeypatch.setenv("APPROVAL_WEBHOOK_URL", "")
    monkeypatch.setenv("REVIEW_ROUNDS", "1")
    monkeypatch.setenv("MAX_EXPERIMENT_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("LOCAL_MODEL_PORT_BASE", "19100")

    # Use local run mode by default for tests.
    monkeypatch.setenv("RUN_MODE", "local")

    s = Settings()
    # Mirror runtime behavior in aras.config.get_settings()
    base = s.local_model_port_base
    s.local_model_ports = (base, base + 1, base + 2)
    return s


@pytest.fixture()
def mock_event_sink() -> list[tuple[str, str, str]]:
    """
    A simple sink for agent emit() events: (agent_id, message, level).
    """
    return []


@pytest.fixture()
def sample_plan() -> dict[str, Any]:
    return {
        "hypothesis": "Test hypothesis for unit testing",
        "questions": ["Q1", "Q2", "Q3"],
        "experiments": ["exp1", "exp2", "exp3"],
        "metrics": ["accuracy", "f1", "runtime"],
        "outline": ["intro", "methodology", "results"],
        "keywords": ["machine learning", "classification", "benchmark"],
        "domain": "general_ml",
    }


@pytest.fixture()
def sample_scraped_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for i in range(5):
        items.append(
            {
                "source": "arxiv",
                "title": f"Test Paper {i}",
                "abstract": "This is a test abstract.",
                "authors": ["A. Author", "B. Author"],
                "published": "2025-01-01",
                "url": f"https://example.com/paper/{i}",
                "doi": f"10.0000/test.{i}",
                "citation_count": 10 + i,
                "has_code": True,
                "code_url": f"https://github.com/example/repo{i}",
                "relevance": 0.9,
                "validated": False,
            }
        )
    return items


@pytest.fixture()
def sample_results_json(tmp_workspace: Path) -> dict[str, Any]:
    exp_dir = tmp_workspace / "experiments" / "test_slug"
    exp_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "runs": [
            {"name": "exp1", "metrics": {"accuracy": 0.8, "losses": [1.0, 0.8, 0.6]}},
            {"name": "exp2", "metrics": {"f1_macro": 0.75, "losses": [1.1, 0.9, 0.7]}},
        ],
        "summary": {"best_accuracy": 0.8},
    }
    (exp_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    return results


@pytest.fixture()
def sample_paper_tex(tmp_workspace: Path) -> str:
    tex = (
        "\\documentclass[conference]{IEEEtran}\n"
        "\\begin{document}\n"
        "\\title{Test Paper}\n"
        "\\maketitle\n"
        "\\begin{abstract}\n"
        "Test abstract.\n"
        "\\end{abstract}\n"
        "\\section{Introduction}\n"
        "Hello.\n"
        "\\section{Methodology}\n"
        "Method.\n"
        "\\section{Results}\n"
        "Results.\n"
        "\\section{Conclusion}\n"
        "Done.\n"
        "\\end{document}\n"
    )
    (tmp_workspace / "paper" / "paper.tex").write_text(tex, encoding="utf-8")
    return tex


@pytest.fixture()
def sample_diff_patch(tmp_workspace: Path) -> tuple[str, dict[str, Any]]:
    diffs = tmp_workspace / "paper" / "diffs"
    diffs.mkdir(parents=True, exist_ok=True)
    patch_text = (
        "--- a/paper.tex\n"
        "+++ b/paper.tex\n"
        "@@ -1,3 +1,3 @@\n"
        "-Old line\n"
        "+New line\n"
    )
    meta = {
        "round": 1,
        "sections_changed": ["introduction"],
        "lines_added": 1,
        "lines_removed": 1,
        "summary": "Replace a line for test.",
    }
    (diffs / "paper_diff_round1.patch").write_text(patch_text, encoding="utf-8")
    (diffs / "round_1_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return patch_text, meta


@pytest.fixture()
def sample_figure_png(tmp_workspace: Path) -> Path:
    # Use Agg backend for headless environments.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = tmp_workspace / "paper" / "figures" / "test.png"
    plt.figure(figsize=(1, 1), dpi=100)
    plt.plot([0, 1], [0, 1])
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


@pytest.fixture()
def mock_llm_response(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """
    Patch FallbackRouter.chat to return a configurable ChatResult.

    Usage:
      mock_llm_response(text='{\"hypothesis\":\"test\"}', tokens=100)
    """

    def _apply(
        *,
        text: str | None = None,
        tokens: int = 100,
        provider: str = "local",
        model: str = "local",
    ) -> None:
        payload = text or json.dumps(
            {
                "hypothesis": "Test hypothesis for unit testing",
                "questions": ["Q1", "Q2", "Q3"],
                "experiments": ["exp1", "exp2", "exp3"],
                "metrics": ["accuracy", "f1", "runtime"],
                "outline": ["intro", "methodology", "results"],
                "keywords": ["machine learning", "classification", "benchmark"],
            },
            ensure_ascii=False,
        )

        async def _chat(self: FallbackRouter, **kwargs: Any) -> ChatResult:  # type: ignore[override]
            ti = tokens // 2
            to = tokens - ti
            return ChatResult(
                text=payload,
                provider=provider,  # type: ignore[arg-type]
                model=model,
                tokens_input=ti,
                tokens_output=to,
                tokens_total=tokens,
                tokens_used=tokens,
            )

        monkeypatch.setattr(FallbackRouter, "chat", _chat, raising=True)

    # Apply default immediately.
    _apply()
    return _apply


@dataclass
class FakeHfApi:
    calls: list[tuple[str, dict[str, Any]]]

    class _SpaceInfo:
        def __init__(self, stage: str) -> None:
            self.stage = stage

    def create_repo(self, **kwargs: Any) -> None:
        self.calls.append(("create_repo", dict(kwargs)))

    def upload_folder(self, **kwargs: Any) -> None:
        self.calls.append(("upload_folder", dict(kwargs)))

    def space_info(self, repo_id: str) -> Any:
        self.calls.append(("space_info", {"repo_id": repo_id}))
        # Immediately look "running" for tests.
        return FakeHfApi._SpaceInfo(stage="RUNNING")


@pytest.fixture()
def fake_hf_api(monkeypatch: pytest.MonkeyPatch) -> FakeHfApi:
    """Monkeypatch huggingface_hub.HfApi to avoid real network calls."""
    api = FakeHfApi(calls=[])

    # Provide a minimal SpaceStage stub, and ensure HuggingFaceAgent imports succeed.
    fake_mod = types.SimpleNamespace(
        HfApi=lambda token=None: api,
        SpaceStage=types.SimpleNamespace(RUNNING="RUNNING"),
        CommitOperationAdd=object,
    )
    monkeypatch.setitem(os.sys.modules, "huggingface_hub", fake_mod)
    return api


@pytest.fixture()
def sink_fn(mock_event_sink: list[tuple[str, str, str]]):
    """Return an EventSink-compatible callable that records emitted events."""

    def _sink(agent_id: str, message: str, *, level: str = "info", **_kw: Any) -> None:
        mock_event_sink.append((str(agent_id), str(message), str(level)))

    return _sink

