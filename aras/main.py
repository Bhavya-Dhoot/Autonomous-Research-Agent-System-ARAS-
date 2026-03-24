from __future__ import annotations

import argparse
import asyncio
import webbrowser
from pathlib import Path

from aras.config import get_settings
from aras.orchestrator import Orchestrator
from aras.approval.gate import ApprovalGate
from aras.ui.server import run_ui
from aras.utils.fs import ensure_dirs
from aras.utils.logging import configure_logging, get_logger


async def _open_browser(url: str) -> None:
    await asyncio.sleep(1.0)
    try:
        webbrowser.open(url, new=2)
    except Exception:
        return


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="aras", description="Autonomous Research Agent System")
    p.add_argument("--topic", required=True, help="Research topic string")
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    p.add_argument("--cycle", type=int, default=1, help="Research cycles to run")
    return p.parse_args()


async def _amain() -> int:
    args = _parse_args()
    settings = get_settings()
    paths = settings.resolved_paths()
    ensure_dirs(list(paths.values()))
    configure_logging(Path(paths["logs_dir"]))
    log = get_logger("main")

    ui_url = f"http://{settings.ui_host}:{settings.ui_port}"
    if not args.no_browser:
        asyncio.create_task(_open_browser(ui_url))

    orchestrator = Orchestrator(settings=settings)
    approval_gate = ApprovalGate(settings=settings)
    orchestrator.approval_gate = approval_gate  # set after init (avoid signature churn)
    ui_task = asyncio.create_task(run_ui(settings=settings, orchestrator=orchestrator, approval_gate=approval_gate))
    await asyncio.sleep(0.5)

    log.info("Starting research cycle(s). topic=%s cycles=%s", args.topic, args.cycle)
    report = await orchestrator.run(topic=args.topic, cycles=args.cycle)

    log.info("Final report: %s", report.to_json(indent=2))
    await orchestrator.shutdown()
    ui_task.cancel()
    return 0


def main() -> int:
    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

