from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aras.agents.base import BaseAgent, EventSink
from aras.config import Settings
from aras.memory.prompt_manager import PromptManager
from aras.memory.rag import RAGContextBuilder
from aras.memory.vector_store import ChromaVectorStore, MemoryDoc


class MemoryAgent(BaseAgent):
    """Persistent memory agent backed by ChromaDB."""

    def __init__(
        self,
        settings: Settings,
        on_event: EventSink,
        on_tokens=None,
        on_chat_result=None,
    ) -> None:
        super().__init__(agent_id="memory", on_event=on_event, on_tokens=on_tokens, on_chat_result=on_chat_result)
        self.settings = settings
        self.store = ChromaVectorStore(settings=settings)
        self.rag = RAGContextBuilder(store=self.store)
        self.prompts = PromptManager(settings=settings)

    async def startup(self) -> None:
        """Ensure all collections exist and prompts initialized."""
        for name in [
            "research_memory",
            "agent_feedback",
            "learned_heuristics",
            "experiment_results",
            "citation_db",
            "failure_db",
            "prompt_versions",
        ]:
            self.store.collection(name)
        pv = self.prompts.latest()
        self._store_prompt_version(pv.version, pv.prompts)
        self.emit(f"Memory ready. prompt_version={pv.version}")

    async def close(self) -> None:
        """Close resources (no-op for persistent chroma client)."""
        return

    async def snapshot(self, *, out_dir: Path, label: str) -> Path | None:
        """Create a local zip snapshot of the persistent Chroma directory.

        This runs independently of GitHub publishing so memory is preserved locally.
        """
        try:
            chroma_dir = Path(self.settings.chroma_persist_dir).resolve()
            if not chroma_dir.exists():
                return None
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            zip_path = out_dir / f"chroma_db_{label}_{ts}.zip"
            await self._zip_dir(src=chroma_dir, out_zip=zip_path)
            self.emit(f"Saved memory snapshot: {zip_path.name}")
            return zip_path
        except Exception as e:
            self.emit(f"Snapshot failed: {e}", level="error")
            return None

    async def _zip_dir(self, *, src: Path, out_zip: Path) -> None:
        def _do() -> None:
            with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for p in src.rglob("*"):
                    if p.is_file():
                        z.write(p, p.relative_to(src))

        import asyncio

        await asyncio.to_thread(_do)

    def current_prompts(self) -> dict[str, str]:
        """Return latest prompts by agent name."""
        return self.prompts.latest().prompts

    def rag_context(self, *, query: str, collections: list[str] | None = None) -> str:
        """Build RAG context for a query."""
        cols = collections or ["learned_heuristics", "agent_feedback", "research_memory", "experiment_results"]
        return self.rag.build(
            query=query,
            collections=cols,
            n=4,
        )

    async def store_cycle(
        self,
        *,
        topic: str,
        plan: dict[str, Any],
        scraped: list[dict[str, Any]],
        results: dict[str, Any],
        analysis: dict[str, Any],
        review: dict[str, Any],
        paper_score: float | None,
    ) -> None:
        """Persist a cycle summary and artifacts into memory collections."""
        ts = datetime.now(timezone.utc).isoformat()
        cycle_id = hashlib.sha256(f"{topic}|{ts}".encode("utf-8")).hexdigest()[:16]

        summary = {
            "topic": topic,
            "ts": ts,
            "plan": plan,
            "scraped_count": len(scraped),
            "results": results.get("summary", results),
            "analysis": analysis,
            "review": review,
            "paper_score": paper_score,
        }

        self.store.upsert(
            collection="research_memory",
            docs=[
                MemoryDoc(
                    id=f"cycle_{cycle_id}",
                    text=json.dumps(summary, ensure_ascii=False),
                    metadata={"topic": topic, "tag": f"cycle_{cycle_id}", "ts": ts},
                )
            ],
        )

        heur = review.get("lessons_learned") or analysis.get("lessons_learned") or ""
        if heur:
            self.store.upsert(
                collection="learned_heuristics",
                docs=[
                    MemoryDoc(
                        id=f"heur_{cycle_id}",
                        text=str(heur),
                        metadata={"topic": topic, "tag": f"heur_{cycle_id}", "ts": ts},
                    )
                ],
            )

        self.store.upsert(
            collection="experiment_results",
            docs=[
                MemoryDoc(
                    id=f"exp_{cycle_id}",
                    text=json.dumps(results, ensure_ascii=False)[:20000],
                    metadata={"topic": topic, "tag": f"exp_{cycle_id}", "ts": ts},
                )
            ],
        )

        self.emit(f"Stored cycle in memory. id={cycle_id}")

    async def store_citations(self, *, citations: list[dict[str, Any]], cycle: int) -> None:
        """Persist validated citations into `citation_db` for later RAG."""
        ts = datetime.now(timezone.utc).isoformat()
        docs: list[MemoryDoc] = []
        for c in citations:
            if not isinstance(c, dict):
                continue
            doi = str(c.get("doi") or "").strip().lower()
            url = str(c.get("url") or "").strip()
            key_src = doi or url or str(c.get("title") or "")
            hid = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]
            cid = f"cite_{hid}_cycle{cycle}"
            text = json.dumps(c, ensure_ascii=False)
            metadata = {
                "tag": "citation_db",
                "cycle": int(cycle),
                "doi": doi or None,
                "source": str(c.get("source") or "crossref"),
                "validated": bool(c.get("validated")),
                "retracted": bool(c.get("retracted")),
                "ts": ts,
            }
            docs.append(MemoryDoc(id=cid, text=text, metadata=metadata))

        if docs:
            self.store.upsert(collection="citation_db", docs=docs)
            self.emit(f"Stored {len(docs)} citations in memory")

    async def store_failure(self, *, failure: dict[str, Any], cycle: int) -> None:
        """Persist a structured failure taxonomy record."""
        ts = datetime.now(timezone.utc).isoformat()
        msg = failure.get("message") or ""
        ftype = str(failure.get("failure_type") or "unknown_error")
        aid = str(failure.get("agent_id") or "unknown")
        ctx = failure.get("context") or {}
        raw_id = f"{aid}|{ftype}|{cycle}|{msg}"
        hid = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
        did = f"fail_{hid}_cycle{cycle}"
        text = json.dumps({"type": ftype, "agent_id": aid, "cycle": cycle, "message": msg, "context": ctx}, ensure_ascii=False)
        self.store.upsert(
            collection="failure_db",
            docs=[
                MemoryDoc(
                    id=did,
                    text=text,
                    metadata={"tag": "failure_db", "cycle": int(cycle), "failure_type": ftype, "agent_id": aid, "ts": ts},
                )
            ],
        )
        self.emit(f"Stored failure in memory. type={ftype} cycle={cycle}")

    def _store_prompt_version(self, version: int, prompts: dict[str, str]) -> None:
        self.store.upsert(
            collection="prompt_versions",
            docs=[
                MemoryDoc(
                    id=f"prompts_v{version}",
                    text=json.dumps(prompts, ensure_ascii=False, indent=2),
                    metadata={"tag": f"prompts_v{version}", "version": version},
                )
            ],
        )

    async def preview(self) -> str:
        """Return a small memory preview for the UI."""
        pv = self.prompts.latest()
        ctx = self.rag.build(query="latest lessons learned", collections=["learned_heuristics"], n=4)
        out = f"prompt_version: {pv.version}\n\n{ctx}"
        return out.strip()

    async def bump_prompts(self, *, updated: dict[str, str]) -> int:
        """Save a new prompt version and persist into vector store."""
        pv = self.prompts.bump(updated_prompts=updated)
        self._store_prompt_version(pv.version, pv.prompts)
        self.emit(f"Saved prompt version v{pv.version}")
        return pv.version
