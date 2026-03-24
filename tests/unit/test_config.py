from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aras.config import Settings


def test_defaults_are_sane() -> None:
    s = Settings()
    assert s.review_rounds == 3
    assert s.budget_usd_ceiling > 0
    assert s.ui_port > 0
    assert s.figure_quality_rerun_enabled is True
    assert s.figure_quality_max_reruns >= 0
    assert 0.0 <= s.novelty_pivot_max_score <= 1.0
    assert 0.0 <= s.novelty_min_confidence <= 1.0
    assert s.novelty_min_validated_evidence >= 1
    assert s.novelty_min_evidence_sources >= 1


def test_missing_optional_keys_dont_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    s = Settings()
    assert s.openai_api_key is None or isinstance(s.openai_api_key, str)


def test_resolved_paths_creates_dirs(tmp_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_workspace)
    s = Settings()
    paths = s.resolved_paths()
    assert {"chroma_persist_dir", "logs_dir", "paper_dir", "experiments_dir", "prompt_store_path"} <= set(paths.keys())


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUDGET_USD_CEILING", "1.0")
    s = Settings()
    assert s.budget_usd_ceiling == 1.0


def test_invalid_budget_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUDGET_USD_CEILING", "-1")
    with pytest.raises(ValidationError):
        Settings()
