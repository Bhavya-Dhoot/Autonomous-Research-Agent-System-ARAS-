from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from aras.agents.base import BaseAgent, EventSink
from aras.config import Settings
from aras.scraping.cache import RedisCache
from aras.utils.logging import get_logger


log = get_logger("citation-validator")


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


def _extract_doi(text: str) -> str | None:
    m = _DOI_RE.search(text)
    if not m:
        return None
    return m.group(0).lower()


def _safe_first_string(x: Any) -> str | None:
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x and isinstance(x[0], str):
        return x[0]
    return None


def _detect_retracted(work: dict[str, Any]) -> bool:
    relations = work.get("relation") or []
    if not isinstance(relations, list):
        return False
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        rtype = str(rel.get("type") or "").lower()
        if "retract" in rtype or "retraction" in rtype:
            return True
    return False


@dataclass(frozen=True)
class CrossrefWork:
    doi: str
    title: str
    url: str
    authors: list[str]
    year: str | None
    citation_count: int
    retracted: bool
    raw: dict[str, Any]


class CitationValidatorAgent(BaseAgent):
    """Validate scraped citations using Crossref and enrich metadata.

    The validator is intentionally non-LLM: it performs DOI/title resolution and
    updates scraped items with DOI, authors, publication year, and retraction flags.
    """

    def __init__(
        self,
        settings: Settings,
        on_event: EventSink,
        on_tokens=None,
        on_chat_result=None,
    ) -> None:
        super().__init__(agent_id="citations", on_event=on_event, on_tokens=on_tokens, on_chat_result=on_chat_result)
        self.settings = settings
        self._cache = RedisCache(settings=settings)
        self._cache_ttl_seconds = int(settings.scrape_cache_ttl_seconds)
        self._email = settings.crossref_email

    async def validate(self, *, items: list[dict[str, Any]], cycle: int) -> list[dict[str, Any]]:
        """Validate and enrich scraped citation metadata."""
        sem = asyncio.Semaphore(5)
        out: list[dict[str, Any]] = []

        async def _one(it: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                return await self._validate_one(it, cycle=cycle)

        tasks = [_one(it) for it in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                self.emit(f"Crossref validation failed: {r}", level="error")
                continue
            out.append(r)
        return out

    async def _validate_one(self, it: dict[str, Any], *, cycle: int) -> dict[str, Any]:
        title = str(it.get("title") or "").strip()
        url = str(it.get("url") or "").strip()
        extra = it.get("extra") or {}
        doi = None
        if isinstance(extra, dict):
            doi = _extract_doi(json.dumps(extra, ensure_ascii=False))
        if not doi:
            doi = _extract_doi(url)

        # If DOI is already present in the item, trust it.
        if not doi:
            doi = _extract_doi(str(it.get("doi") or ""))

        if doi:
            work = await self._lookup_by_doi(doi)
        else:
            work = await self._lookup_by_title(title)

        if not work:
            it["validated"] = False
            it["retracted"] = bool(it.get("retracted"))
            it["crossref"] = None
            return it

        it["doi"] = work.doi
        it["validated"] = True
        it["retracted"] = work.retracted
        it["citation_count"] = work.citation_count
        it["published"] = work.year or it.get("published")
        if work.title:
            it["title"] = work.title
        if work.url:
            it["url"] = work.url
        it["authors"] = work.authors or it.get("authors") or []

        # Attach raw info for debugging and later use.
        it["crossref"] = work.raw

        return it

    def _doi_cache_key(self, doi: str) -> str:
        return f"crossref:doi:{doi}"

    def _title_cache_key(self, title: str) -> str:
        t = re.sub(r"\s+", " ", title.strip().lower())[:140]
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()
        return f"crossref:title:{h}"

    async def _lookup_by_doi(self, doi: str) -> CrossrefWork | None:
        cache_key = self._doi_cache_key(doi)
        cached = self._cache.get(cache_key)
        if cached:
            try:
                obj = json.loads(cached)
                return self._work_from_json(obj)
            except Exception:
                pass

        params: dict[str, str] = {}
        if self._email:
            params["mailto"] = self._email
        headers = {"User-Agent": "ARAS/1.0"}

        url = f"https://api.crossref.org/works/{doi}"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(url, params=params, headers=headers)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                payload = r.json()
        except Exception as e:
            self.emit(f"Crossref DOI lookup failed for {doi}: {e}", level="error")
            return None

        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, dict):
            return None

        try:
            self._cache.set(cache_key, json.dumps(message, ensure_ascii=False))
        except Exception:
            pass
        return self._work_from_json(message)

    async def _lookup_by_title(self, title: str) -> CrossrefWork | None:
        if not title:
            return None

        cache_key = self._title_cache_key(title)
        cached = self._cache.get(cache_key)
        if cached:
            try:
                obj = json.loads(cached)
                return self._work_from_json(obj)
            except Exception:
                pass

        params: dict[str, str] = {
            "query.bibliographic": title,
            "rows": "1",
        }
        if self._email:
            params["mailto"] = self._email
        headers = {"User-Agent": "ARAS/1.0"}

        url = "https://api.crossref.org/works"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(url, params=params, headers=headers)
                r.raise_for_status()
                payload = r.json()
        except Exception as e:
            self.emit(f"Crossref title lookup failed: {e}", level="error")
            return None

        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, dict):
            return None

        items = message.get("items") or []
        if not isinstance(items, list) or not items:
            return None
        work = items[0]
        if not isinstance(work, dict):
            return None

        try:
            self._cache.set(cache_key, json.dumps(work, ensure_ascii=False))
        except Exception:
            pass
        return self._work_from_json(work)

    def _work_from_json(self, work: dict[str, Any]) -> CrossrefWork | None:
        doi = str(work.get("DOI") or "").strip().lower()
        if not doi:
            return None

        title_list = work.get("title") or []
        title = _safe_first_string(title_list) or str(work.get("alternative-id") or "").strip()

        url = str(_safe_first_string(work.get("URL")) or "").strip()
        if not url:
            # Crossref sometimes provides URL in DOI domain.
            url = f"https://doi.org/{doi}"

        authors: list[str] = []
        for a in work.get("author") or []:
            if not isinstance(a, dict):
                continue
            given = str(a.get("given") or "").strip()
            family = str(a.get("family") or "").strip()
            full = " ".join([x for x in [given, family] if x]).strip()
            if full:
                authors.append(full)

        year: str | None = None
        issued = work.get("issued") or {}
        if isinstance(issued, dict):
            parts = issued.get("date-parts") or []
            if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
                year = str(parts[0][0])
        citation_count = int(work.get("is-referenced-by-count") or 0)
        retracted = _detect_retracted(work)

        return CrossrefWork(
            doi=doi,
            title=title or "",
            url=url,
            authors=authors,
            year=year,
            citation_count=citation_count,
            retracted=retracted,
            raw=work,
        )

