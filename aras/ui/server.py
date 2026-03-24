from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

from aras.approval.gate import ApprovalGate
from aras.config import Settings
from aras.types import ApprovalDecision, ApprovalPayload
from aras.utils.json import dumps
from aras.utils.logging import get_logger


log = get_logger("ui")


@dataclass
class PendingApproval:
    request_id: str
    payload: ApprovalPayload
    future: asyncio.Future[ApprovalDecision]
    created_at: datetime


class BroadcastHub:
    """Broadcast state/log events to all connected websockets."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        data = dumps(payload)
        async with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_text(data)
            except Exception:
                await self.disconnect(ws)


async def _read_json_request_body(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_metric_event(message: str) -> Optional[dict[str, Any]]:
    if not message.startswith("METRIC_EVENT "):
        return None
    raw = message[len("METRIC_EVENT ") :].strip()
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None
        # Normalize to required shape.
        exp = obj.get("experiment")
        key = obj.get("key")
        val = obj.get("value")
        step = obj.get("step")
        if exp is None or key is None or val is None:
            return None
        return {"experiment": exp, "key": key, "value": val, "step": step}
    except Exception:
        return None


def _parse_hf_event(message: str) -> Optional[dict[str, Any]]:
    if not message.startswith("HF_EVENT "):
        return None
    raw = message[len("HF_EVENT ") :].strip()
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def _parse_diff_metadata(patch_text: str) -> tuple[list[str], int, int]:
    # Best-effort: infer current section from last \section{...} seen in patch while counting +/-
    sec_re = re.compile(r"\\section\{([^}]+)\}")
    # Include headings if present.
    current: str | None = None
    changed_sections: set[str] = set()
    added = 0
    removed = 0

    for line in patch_text.splitlines():
        if line.startswith("@@"):
            continue
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        # Keep section tracking on context lines too.
        if "\\section{" in line or "\\subsection{" in line:
            m = sec_re.search(line)
            if m:
                current = m.group(1).strip().lower()
        # Count diff changes.
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
            if current:
                changed_sections.add(current)
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
            if current:
                changed_sections.add(current)
        else:
            # context lines " " still may carry current section info
            if current and ("\\section{" in line):
                m2 = sec_re.search(line)
                if m2:
                    current = m2.group(1).strip().lower()

    # Normalize common section names.
    norm: list[str] = []
    for s in sorted(changed_sections):
        s2 = s.replace(" ", "_")
        norm.append(s2)
    return norm, added, removed


def _safe_filename_to_round(filename: str) -> int | None:
    m = re.search(r"paper_diff_round(\d+)\.patch$", filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


class UiDiffFigureManager:
    """Scans disk and broadcasts figure/diff readiness events."""

    def __init__(self, *, hub: BroadcastHub, paper_dir: Path) -> None:
        self.hub = hub
        self.paper_dir = paper_dir
        self.diff_dir = paper_dir / "diffs"
        self.fig_dir = paper_dir / "figures"
        self._announced_diffs: set[str] = set()
        self._announced_figs: set[str] = set()

    async def watch_diffs_loop(self, *, interval_s: float = 5.0, shutdown_event: asyncio.Event) -> None:
        while not shutdown_event.is_set():
            await asyncio.sleep(interval_s)
            try:
                if not self.diff_dir.exists():
                    continue
                for patch in sorted(self.diff_dir.glob("paper_diff_round*.patch"), key=lambda p: p.stat().st_mtime):
                    fname = patch.name
                    if fname in self._announced_diffs:
                        continue
                    round_n = _safe_filename_to_round(fname)
                    if round_n is None:
                        continue
                    patch_text = patch.read_text(encoding="utf-8")
                    sections_changed, added, removed = _parse_diff_metadata(patch_text)
                    meta_path = self.diff_dir / f"round_{round_n}_meta.json"
                    summary = None
                    if meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text(encoding="utf-8"))
                            if isinstance(meta, dict) and isinstance(meta.get("summary"), str):
                                summary = meta["summary"]
                            if isinstance(meta.get("sections_changed"), list) and meta["sections_changed"]:
                                sections_changed = [str(x) for x in meta["sections_changed"]]
                            if isinstance(meta.get("lines_added"), int):
                                added = meta["lines_added"]
                            if isinstance(meta.get("lines_removed"), int):
                                removed = meta["lines_removed"]
                        except Exception:
                            summary = None
                    if not summary:
                        summary = f"Auto diff: +{added} lines, -{removed} lines."

                    await self.hub.broadcast(
                        {
                            "type": "diff",
                            "round": round_n,
                            "sections_changed": sections_changed,
                            "lines_added": added,
                            "lines_removed": removed,
                            "patch": patch_text,
                            "summary": summary,
                        }
                    )
                    self._announced_diffs.add(fname)
            except Exception:
                # best-effort watcher
                pass

    async def watch_figures_loop(self, *, interval_s: float = 5.0, shutdown_event: asyncio.Event) -> None:
        while not shutdown_event.is_set():
            await asyncio.sleep(interval_s)
            try:
                if not self.fig_dir.exists():
                    continue
                for png in sorted(self.fig_dir.rglob("*.png"), key=lambda p: p.stat().st_mtime):
                    rel = png.relative_to(self.fig_dir)
                    # Use relative path as unique id.
                    key = str(rel)
                    if key in self._announced_figs:
                        continue
                    name = png.stem
                    # Caption: best-effort from sibling captions.json (if present).
                    caption = ""
                    cap_path = png.parent / "captions.json"
                    if cap_path.exists():
                        try:
                            caps = json.loads(cap_path.read_text(encoding="utf-8"))
                            if isinstance(caps, dict):
                                c_obj = caps.get(name) or caps.get(key) or ""
                                if isinstance(c_obj, dict):
                                    caption = str(c_obj.get("caption") or "")
                                else:
                                    caption = str(c_obj or "")
                        except Exception:
                            caption = ""
                    if not caption:
                        caption = name
                    experiment = png.parent.name if png.parent != self.fig_dir else "experiment"
                    png_url = f"/paper-figures/{str(rel).replace('\\\\','/')}"

                    await self.hub.broadcast(
                        {
                            "type": "figure_ready",
                            "name": name,
                            "png_url": png_url,
                            "caption": caption,
                            "experiment": experiment,
                            "fig_type": "figure",
                        }
                    )
                    self._announced_figs.add(key)
            except Exception:
                pass


def build_app(*, index_html: str, hub: BroadcastHub, orchestrator: Any, settings: Settings, approval_gate: ApprovalGate | None = None) -> FastAPI:
    app = FastAPI()

    pending_lock = asyncio.Lock()
    pending_by_request_id: dict[str, PendingApproval] = {}

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(index_html)

    # Existing approval integration (optional). UI-driven local approvals use /api/approval/webhook + /api/approval/decision.
    @app.post("/api/approval/request")
    async def api_approval_request(payload: ApprovalPayload) -> ApprovalDecision:
        if approval_gate is None:
            return ApprovalDecision(approved=True, note="auto-approved (no ApprovalGate configured)")
        return await approval_gate.request(payload=payload)

    # Backwards-compatible alias.
    @app.post("/api/approval")
    async def api_approval(payload: ApprovalPayload) -> ApprovalDecision:
        return await api_approval_request(payload)

    @app.post("/api/approval/webhook")
    async def api_approval_webhook(request: Request) -> ApprovalDecision:
        body = await _read_json_request_body(request)
        request_id = str(body.get("request_id") or "")
        if not request_id:
            return ApprovalDecision(approved=False, note="missing request_id")
        # Remove request_id before constructing the pydantic model.
        payload_dict = dict(body)
        payload_dict.pop("request_id", None)
        try:
            payload = ApprovalPayload(**payload_dict)
        except Exception:
            return ApprovalDecision(approved=False, note="invalid approval payload")

        fut: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
        async with pending_lock:
            pending_by_request_id[request_id] = PendingApproval(
                request_id=request_id,
                payload=payload,
                future=fut,
                created_at=datetime.now(timezone.utc),
            )

        # Block until decision is posted or timeout expires.
        try:
            timeout_s = float(settings.approval_timeout_seconds)
            decision = await asyncio.wait_for(fut, timeout=timeout_s)
            return decision
        except asyncio.TimeoutError:
            return ApprovalDecision(approved=False, note="approval timeout")
        finally:
            async with pending_lock:
                pending_by_request_id.pop(request_id, None)

    @app.post("/api/approval/decision")
    async def api_approval_decision(request: Request) -> ApprovalDecision:
        body = await _read_json_request_body(request)
        request_id = str(body.get("request_id") or "")
        if not request_id:
            return ApprovalDecision(approved=False, note="missing request_id")
        approved_raw = body.get("approved")
        note = body.get("note")
        approved = bool(approved_raw) if approved_raw is not None else False

        async with pending_lock:
            pending = pending_by_request_id.get(request_id)
            if not pending:
                return ApprovalDecision(approved=False, note="no such pending request")
            if pending.future.done():
                return ApprovalDecision(approved=bool(pending.future.result().approved), note="already decided")
            pending.future.set_result(ApprovalDecision(approved=approved, note=note))

        return ApprovalDecision(approved=approved, note=note)

    @app.get("/api/approval/pending")
    async def api_approval_pending() -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        async with pending_lock:
            if not pending_by_request_id:
                return {"pending": False}
            # Only expose the most recent pending request.
            items = list(pending_by_request_id.values())
            items.sort(key=lambda p: p.created_at, reverse=True)
            p = items[0]
            if not p.request_id:
                return {"pending": False}
            expires_in = max(0.0, float(settings.approval_timeout_seconds) - (now - p.created_at).total_seconds())
            return {
                "pending": True,
                "request_id": p.request_id,
                "expires_in_seconds": expires_in,
                "payload": p.payload.model_dump(),
            }

    @app.get("/api/diffs")
    async def api_diffs() -> list[dict[str, Any]]:
        diffs_dir = Path("paper") / "diffs"
        if not diffs_dir.exists():
            return []

        out: list[dict[str, Any]] = []
        for patch in diffs_dir.glob("paper_diff_round*.patch"):
            round_n = _safe_filename_to_round(patch.name)
            if round_n is None:
                continue
            patch_text = patch.read_text(encoding="utf-8", errors="replace")
            sections_changed, added, removed = _parse_diff_metadata(patch_text)
            meta_path = diffs_dir / f"round_{round_n}_meta.json"
            summary = None
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if isinstance(meta, dict):
                        if isinstance(meta.get("summary"), str):
                            summary = meta["summary"]
                        if isinstance(meta.get("sections_changed"), list) and meta["sections_changed"]:
                            sections_changed = [str(x) for x in meta["sections_changed"]]
                        if isinstance(meta.get("lines_added"), int):
                            added = int(meta["lines_added"])
                        if isinstance(meta.get("lines_removed"), int):
                            removed = int(meta["lines_removed"])
                except Exception:
                    summary = None
            if not summary:
                summary = f"Auto diff: +{added} lines, -{removed} lines."
            out.append(
                {
                    "round": round_n,
                    "filename": patch.name,
                    "sections_changed": sections_changed,
                    "lines_added": added,
                    "lines_removed": removed,
                    "summary": summary,
                }
            )

        out.sort(key=lambda x: int(x["round"]))
        return out

    @app.get("/api/figures")
    async def api_figures() -> list[dict[str, Any]]:
        fig_dir = Path("paper") / "figures"
        if not fig_dir.exists():
            return []
        out: list[dict[str, Any]] = []
        for png in fig_dir.rglob("*.png"):
            rel = png.relative_to(fig_dir)
            name = png.stem
            png_url = f"/paper-figures/{str(rel).replace('\\\\','/')}"
            caption = ""
            cap_path = png.parent / "captions.json"
            if cap_path.exists():
                try:
                    caps = json.loads(cap_path.read_text(encoding="utf-8"))
                    if isinstance(caps, dict):
                        c_obj = caps.get(name) or ""
                        if isinstance(c_obj, dict):
                            caption = str(c_obj.get("caption") or "")
                        else:
                            caption = str(c_obj or "")
                except Exception:
                    caption = ""
            if not caption:
                caption = name
            experiment = rel.parts[0] if len(rel.parts) > 1 else "experiment"
            out.append(
                {
                    "name": name,
                    "png_url": png_url,
                    "caption": caption,
                    "experiment": experiment,
                    "type": "figure",
                }
            )
        out.sort(key=lambda x: x["name"])
        return out

    @app.get("/api/hf")
    async def api_hf() -> dict[str, Any]:
        """Return last Hugging Face publish URLs (best-effort)."""
        p = Path("logs") / "hf_last.json"
        if not p.exists():
            return {"dataset_url": None, "model_url": None, "space_url": None}
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                return {"dataset_url": None, "model_url": None, "space_url": None}
            return {
                "dataset_url": obj.get("dataset_url"),
                "model_url": obj.get("model_url"),
                "space_url": obj.get("space_url"),
            }
        except Exception:
            return {"dataset_url": None, "model_url": None, "space_url": None}

    @app.get("/api/quality")
    async def api_quality() -> dict[str, Any]:
        """Return latest cycle quality summary from logs/cycle_quality.jsonl."""
        p = Path("logs") / "cycle_quality.jsonl"
        if not p.exists():
            return {"latest": None, "history_count": 0}

        rows: list[dict[str, Any]] = []
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                t = line.strip()
                if not t:
                    continue
                try:
                    obj = json.loads(t)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
        except Exception:
            return {"latest": None, "history_count": 0}

        if not rows:
            return {"latest": None, "history_count": 0}

        return {"latest": rows[-1], "history_count": len(rows)}

    @app.get("/paper-figures/{path:path}")
    async def serve_paper_figures(path: str) -> FileResponse:
        fig_path = Path("paper") / "figures" / Path(path)
        return FileResponse(str(fig_path))

    # Backwards-compatible alias (some UIs might request /figures/).
    @app.get("/figures/{path:path}")
    async def serve_figures_alias(path: str) -> FileResponse:
        fig_path = Path("paper") / "figures" / Path(path)
        return FileResponse(str(fig_path))

    @app.websocket("/ws/status")
    async def ws_status(ws: WebSocket) -> None:
        await hub.connect(ws)
        try:
            while True:
                # Always publish a full UI state snapshot.
                await hub.broadcast({"type": "state", "state": orchestrator.ui_state()})
                # Best-effort cost event payload for COSTS panel.
                st = orchestrator.ui_state()
                await hub.broadcast(
                    {
                        "type": "cost",
                        "total_usd": st.get("cost_usd") or 0.0,
                        "by_agent": {},
                        "by_provider": {},
                        "budget_remaining": st.get("budget_remaining_usd"),
                    }
                )
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            await hub.disconnect(ws)
        except Exception:
            await hub.disconnect(ws)

    return app


async def _pump_logs(hub: BroadcastHub, orchestrator: Any) -> None:
    async for evt in orchestrator.ui_logs():
        try:
            msg = str(evt.get("message") or "")
            metric = _parse_metric_event(msg)
            if metric:
                await hub.broadcast({"type": "metric", **metric})
                continue
            hf_evt = _parse_hf_event(msg)
            if hf_evt:
                await hub.broadcast({"type": "hf_event", **hf_evt})
                continue
            await hub.broadcast({"type": "log", "event": evt})
        except Exception:
            # best-effort; never block UI
            continue


async def run_ui(settings: Settings, orchestrator: Any, approval_gate: ApprovalGate | None = None) -> None:
    """Run the UI server."""
    index_path = Path(__file__).parent / "static" / "index.html"
    index_html = index_path.read_text(encoding="utf-8")
    hub = BroadcastHub()
    app = build_app(
        index_html=index_html,
        hub=hub,
        orchestrator=orchestrator,
        settings=settings,
        approval_gate=approval_gate,
    )

    shutdown_event = asyncio.Event()
    paper_dir = Path("paper").resolve()
    mgr = UiDiffFigureManager(hub=hub, paper_dir=paper_dir)

    asyncio.create_task(_pump_logs(hub, orchestrator))
    asyncio.create_task(mgr.watch_diffs_loop(shutdown_event=shutdown_event))
    asyncio.create_task(mgr.watch_figures_loop(shutdown_event=shutdown_event))

    config = uvicorn.Config(app, host=settings.ui_host, port=settings.ui_port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
