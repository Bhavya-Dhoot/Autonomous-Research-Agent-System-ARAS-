from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from aras.config import Settings
from aras.utils.logging import get_logger


log = get_logger("vector-store")


@dataclass
class MemoryDoc:
    """A memory document stored in the vector DB."""

    id: str
    text: str
    metadata: dict[str, Any]


class ChromaVectorStore:
    """ChromaDB wrapper with named collections."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        persist = Path(settings.chroma_persist_dir).resolve()
        persist.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist))
        self._collections: dict[str, Collection] = {}

    def collection(self, name: str) -> Collection:
        """Get or create a collection."""
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(name=name)
        return self._collections[name]

    def upsert(self, *, collection: str, docs: list[MemoryDoc]) -> None:
        """Upsert docs into a collection."""
        c = self.collection(collection)
        c.upsert(
            ids=[d.id for d in docs],
            documents=[d.text for d in docs],
            metadatas=[d.metadata for d in docs],
        )

    def query(self, *, collection: str, text: str, n: int = 6) -> list[dict[str, Any]]:
        """Query similar docs."""
        c = self.collection(collection)
        res = c.query(query_texts=[text], n_results=int(n))
        out: list[dict[str, Any]] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0] if "distances" in res else [None] * len(ids)
        for i in range(len(ids)):
            out.append({"id": ids[i], "text": docs[i], "metadata": metas[i], "distance": dists[i]})
        return out

