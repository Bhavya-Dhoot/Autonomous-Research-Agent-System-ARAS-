from __future__ import annotations

from typing import Any

from aras.memory.vector_store import ChromaVectorStore


class RAGContextBuilder:
    """Build retrieval-augmented context for prompts."""

    def __init__(self, store: ChromaVectorStore) -> None:
        self.store = store

    def build(self, *, query: str, collections: list[str], n: int = 4) -> str:
        """Return a compact context string from memory collections."""
        chunks: list[str] = []
        for c in collections:
            hits = self.store.query(collection=c, text=query, n=n)
            if not hits:
                continue
            chunks.append(f"[{c}]")
            for h in hits[:n]:
                meta = h.get("metadata") or {}
                tag = meta.get("tag") or meta.get("topic") or h.get("id")
                txt = (h.get("text") or "").strip()
                if len(txt) > 900:
                    txt = txt[:900] + "..."
                chunks.append(f"- {tag}: {txt}")
        return "\n".join(chunks).strip()

