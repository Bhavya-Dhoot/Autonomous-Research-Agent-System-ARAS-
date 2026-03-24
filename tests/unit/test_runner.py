from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from aras.experiments.runner import run_experiment_module


@pytest.mark.asyncio
async def test_runs_trivial_script(tmp_path: Path) -> None:
    p = tmp_path / "exp.py"
    p.write_text('print("hello")\n', encoding="utf-8")
    rr = await run_experiment_module(module_path=p, workdir=tmp_path, timeout_seconds=10)
    assert rr.exit_code == 0
    assert "hello" in rr.stdout


@pytest.mark.asyncio
async def test_parses_metric_json_lines(tmp_path: Path) -> None:
    p = tmp_path / "exp.py"
    p.write_text(
        'print(\'METRIC_JSON {"experiment":"e1","metric":"loss","value":0.5,"step":1}\', flush=True)\n'
        'open("results.json","w",encoding="utf-8").write("{}")\n',
        encoding="utf-8",
    )
    got = []

    def on_metric(obj):
        got.append(obj)

    rr = await run_experiment_module(module_path=p, workdir=tmp_path, timeout_seconds=10, on_metric=on_metric)
    assert rr.exit_code == 0
    assert got and got[0]["experiment"] in {"e1", "exp"}
    assert got[0]["key"] == "loss"
    assert float(got[0]["value"]) == 0.5


@pytest.mark.asyncio
async def test_parses_legacy_metric_format(tmp_path: Path) -> None:
    p = tmp_path / "exp.py"
    p.write_text(
        'print("METRIC epoch=1 loss=0.5", flush=True)\n'
        'open("results.json","w",encoding="utf-8").write("{}")\n',
        encoding="utf-8",
    )
    got = []

    def on_metric(obj):
        got.append(obj)

    rr = await run_experiment_module(module_path=p, workdir=tmp_path, timeout_seconds=10, on_metric=on_metric)
    assert rr.exit_code == 0
    assert got
    assert got[0]["key"] == "loss"
    assert float(got[0]["value"]) == 0.5


@pytest.mark.asyncio
async def test_timeout_kills_process(tmp_path: Path) -> None:
    p = tmp_path / "exp.py"
    p.write_text("while True:\n  pass\n", encoding="utf-8")
    rr = await run_experiment_module(module_path=p, workdir=tmp_path, timeout_seconds=1)
    # exit_code is best-effort; the key assertion is that the call returns.
    assert rr.wall_seconds < 10

