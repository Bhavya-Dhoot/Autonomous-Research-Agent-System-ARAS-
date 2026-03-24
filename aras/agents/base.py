from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol

from aras.utils.logging import get_logger


class EventSink(Protocol):
    """Protocol for sending UI events."""

    def __call__(self, agent: str, message: str, *, level: str = "info") -> None: ...


log = get_logger("agent")


@dataclass
class StructuredError:
    """Structured error object for fault tolerance and logging."""

    agent: str
    kind: str
    message: str
    ts: str
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "kind": self.kind,
            "message": self.message,
            "ts": self.ts,
            "detail": self.detail,
        }


class BaseAgent:
    """Base class for all agents with shared error handling."""

    def __init__(
        self,
        *,
        agent_id: str,
        on_event: EventSink,
        on_tokens: Callable[[int], None] | None = None,
        on_chat_result: Callable[[Any], None] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.on_event = on_event
        self._on_tokens = on_tokens
        self._on_chat_result = on_chat_result
        self._lock = asyncio.Lock()

    def emit(self, message: str, *, level: str = "info") -> None:
        self.on_event(self.agent_id, message, level=level)

    def add_tokens(self, n: int) -> None:
        """Record token usage for UI accounting."""
        if not self._on_tokens:
            return
        try:
            self._on_tokens(int(n))
        except Exception:
            return

    def record_chat_result(self, result: Any) -> None:
        """Record a full LLM call result (used for cost/token aggregation)."""
        if not self._on_chat_result:
            return
        try:
            self._on_chat_result(result)
        except Exception:
            return

    async def guarded(self, kind: str, fn: Callable[[], Awaitable[Any]]) -> Any:
        """Run a coroutine and return structured error on failure."""
        try:
            async with self._lock:
                return await fn()
        except Exception as e:
            err = StructuredError(
                agent=self.agent_id,
                kind=kind,
                message=str(e),
                ts=datetime.now(timezone.utc).isoformat(),
                detail={"type": type(e).__name__},
            )
            self.emit(f"{kind} failed: {err.message}", level="error")
            return err.to_dict()

