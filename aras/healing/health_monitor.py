from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

from aras.config import Settings
from aras.models.local_model_server import LocalModelServerManager


class HealthMonitor:
    """Heartbeat monitor that restarts local model servers best-effort."""

    def __init__(self, settings: Settings, on_event: Callable[[str, str], None]) -> None:
        self.settings = settings
        self.on_event = on_event
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._servers = LocalModelServerManager(settings=settings)

    async def start(self, *, agents: dict[str, Any]) -> None:
        """Start background health monitoring."""
        self._stop.clear()
        await self._servers.start_all()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(agents))

    async def stop(self) -> None:
        """Stop monitoring."""
        self._stop.set()
        if self._task:
            self._task.cancel()
        await self._servers.stop_all()

    async def _run(self, agents: dict[str, Any]) -> None:
        last = time.time()
        while not self._stop.is_set():
            await asyncio.sleep(self.settings.heartbeat_interval_seconds)
            now = time.time()
            if now - last >= self.settings.heartbeat_interval_seconds:
                self.on_event("orchestrator", "heartbeat")
                last = now
            # Best-effort: ensure local model servers are up.
            await self._servers.start_all()

