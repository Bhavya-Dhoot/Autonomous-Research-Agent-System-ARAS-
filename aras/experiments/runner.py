from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import psutil


@dataclass
class ExperimentRunResult:
    name: str
    exit_code: int
    wall_seconds: float
    peak_rss_mb: float
    stdout: str
    stderr: str
    artifacts: dict[str, str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "exit_code": self.exit_code,
            "wall_seconds": self.wall_seconds,
            "peak_rss_mb": self.peak_rss_mb,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
        }


async def run_experiment_module(
    *,
    module_path: Path,
    workdir: Path,
    env: dict[str, str] | None = None,
    timeout_seconds: int = 600,
    on_metric: Callable[[dict[str, Any]], None] | None = None,
    metric_prefix: str = "METRIC_JSON ",
) -> ExperimentRunResult:
    """Run an experiment module in a subprocess and capture peak RSS."""
    # Best-effort auto-install for explicitly allowed missing packages.
    # This runs before executing the subprocess, and only attempts known-safe packages.
    ALLOWED_INSTALL_IMPORT_TO_PIP: dict[str, str] = {
        "nltk": "nltk",
        "gymnasium": "gymnasium",
        "gym": "gym",
        "datasets": "datasets",
        "torchvision": "torchvision",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "seaborn": "seaborn",
        "pandas": "pandas",
        "numpy": "numpy",
    }

    def _parse_import_roots(src_text: str) -> list[str]:
        roots: set[str] = set()
        for line in src_text.splitlines():
            line2 = line.strip()
            # Only consider top-level import lines to avoid false positives.
            if not (line2.startswith("import ") or line2.startswith("from ")):
                continue
            # Skip relative imports.
            if line2.startswith("import "):
                # import X [as Y]
                m = re.match(r"^import\s+([a-zA-Z0-9_\.]+)(?:\s+as\s+[a-zA-Z0-9_]+)?", line2)
                if m:
                    roots.add(m.group(1).split(".")[0])
            if line2.startswith("from "):
                m = re.match(r"^from\s+([a-zA-Z0-9_\.]+)\s+import\s+", line2)
                if m:
                    mod = m.group(1)
                    if mod.startswith("."):
                        continue
                    roots.add(mod.split(".")[0])
        return list(roots)

    def _maybe_auto_install(module_path_: Path) -> None:
        try:
            src = module_path_.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        roots = _parse_import_roots(src)
        for mod_root in roots:
            pip_pkg = ALLOWED_INSTALL_IMPORT_TO_PIP.get(mod_root)
            if not pip_pkg:
                continue
            try:
                if importlib.util.find_spec(mod_root) is not None:
                    continue
            except Exception:
                # If find_spec itself fails, still try best-effort install.
                pass
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", pip_pkg, "-q"], check=False)
            except Exception:
                # Security constraint: never install unknown packages, but allowed ones can fail gracefully.
                pass

    _maybe_auto_install(module_path)

    env2 = dict(os.environ)
    if env:
        env2.update(env)
    name = module_path.stem
    cmd = [sys.executable, str(module_path)]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(workdir),
        env=env2,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    p = psutil.Process(proc.pid)
    start = asyncio.get_event_loop().time()
    peak = 0

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    async def monitor() -> None:
        nonlocal peak
        while proc.returncode is None:
            try:
                rss = p.memory_info().rss
                peak = max(peak, rss)
            except Exception:
                pass
            await asyncio.sleep(0.05)

    mon = asyncio.create_task(monitor())

    async def read_stdout() -> None:
        if proc.stdout is None:
            return
        while True:
            line_b = await proc.stdout.readline()
            if not line_b:
                return
            line = line_b.decode("utf-8", errors="replace")
            stdout_chunks.append(line)
            if not on_metric:
                continue

            if line.startswith(metric_prefix):
                payload = line[len(metric_prefix) :].strip()
                try:
                    obj = json.loads(payload)
                    if not isinstance(obj, dict):
                        continue
                    exp = obj.get("experiment") or name
                    # Spec uses `metric`; UI event wants `key`.
                    key = obj.get("metric") or obj.get("key") or "metric"
                    value = obj.get("value")
                    step = obj.get("step") if obj.get("step") is not None else 0
                    on_metric({"experiment": exp, "key": key, "value": value, "step": step})
                except Exception:
                    pass
                continue

            if line.startswith("METRIC "):
                # Legacy format: METRIC epoch={N} {key}={value}
                # We normalize into {experiment, key, value, step}.
                try:
                    pairs = re.findall(r"(\w+)=([^\s]+)", line)
                    kv = {k: v for (k, v) in pairs}
                    # Infer step from epoch/step
                    step_raw = kv.get("epoch") or kv.get("step") or "0"
                    try:
                        step = int(float(step_raw))
                    except Exception:
                        step = 0
                    # Choose the first non-step field as metric key.
                    key = None
                    value = None
                    for k, v in kv.items():
                        if k in {"epoch", "step"}:
                            continue
                        key = k
                        try:
                            value = float(v)
                        except Exception:
                            # If value isn't numeric, skip normalization.
                            value = None
                        break
                    if key is None or value is None:
                        continue
                    on_metric({"experiment": name, "key": key, "value": value, "step": step})
                except Exception:
                    pass

    async def read_stderr() -> None:
        if proc.stderr is None:
            return
        while True:
            line_b = await proc.stderr.readline()
            if not line_b:
                return
            line = line_b.decode("utf-8", errors="replace")
            stderr_chunks.append(line)

    t_stdout = asyncio.create_task(read_stdout())
    t_stderr = asyncio.create_task(read_stderr())
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
    finally:
        # Ensure reader tasks complete and stop monitoring.
        mon.cancel()
        await asyncio.gather(t_stdout, t_stderr, return_exceptions=True)

    end = asyncio.get_event_loop().time()
    out = "".join(stdout_chunks)
    err = "".join(stderr_chunks)

    metrics_path = workdir / "results.json"
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    artifacts: dict[str, str] = {}
    for ext in ("png", "pdf", "json"):
        for f in workdir.glob(f"*.{ext}"):
            artifacts[f.name] = str(f.resolve())

    return ExperimentRunResult(
        name=name,
        exit_code=int(proc.returncode or 0),
        wall_seconds=float(end - start),
        peak_rss_mb=float(peak / (1024 * 1024)),
        stdout=out[-8000:],
        stderr=err[-8000:],
        artifacts=artifacts,
        metrics=metrics,
    )

