from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from aras.config import Settings
from aras.types import ApprovalDecision, ApprovalPayload
from aras.utils.logging import get_logger


log = get_logger("approval-gate")


@dataclass
class ApprovalRequest:
    request_id: str
    payload: ApprovalPayload


class ApprovalGate:
    """Human-in-the-loop approval gate.

    If `APPROVAL_WEBHOOK_URL` is configured, this gate calls the webhook and
    waits for an `ApprovalDecision`.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._inflight: dict[str, ApprovalRequest] = {}
        self._lock = asyncio.Lock()

    async def request(self, *, payload: ApprovalPayload) -> ApprovalDecision:
        """Request approval for the given payload and return a decision."""
        # If no webhook is configured, keep the system runnable.
        if not self.settings.approval_webhook_url:
            return ApprovalDecision(approved=True, note="auto-approved (no APPROVAL_WEBHOOK_URL configured)")

        request_id = str(uuid.uuid4())
        req = ApprovalRequest(request_id=request_id, payload=payload)
        async with self._lock:
            self._inflight[request_id] = req

        try:
            timeout_s = int(self.settings.approval_timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await client.post(
                    self.settings.approval_webhook_url,
                    json={**payload.model_dump(), "request_id": request_id},
                )
                resp.raise_for_status()
                data: Any = resp.json()
                if isinstance(data, dict) and "approved" in data:
                    return ApprovalDecision(approved=bool(data.get("approved")), note=data.get("note"))
                return ApprovalDecision(approved=False, note="invalid approval response shape")
        except Exception as e:
            log.warning("Approval webhook failed: %s", e)
            return ApprovalDecision(approved=False, note=f"approval webhook error: {e}")
        finally:
            async with self._lock:
                self._inflight.pop(request_id, None)

