from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup

from aras.agents.base import BaseAgent, EventSink
from aras.agents.memory_agent import MemoryAgent
from aras.config import Settings
from aras.scraping.scrape_router import ScrapeRouter
from aras.scraping.parsers.arxiv_parser import parse_arxiv
from aras.scraping.parsers.github_parser import parse_github
from aras.scraping.parsers.generic_parser import parse_generic
from aras.scraping.parsers.paperswithcode_parser import parse_paperswithcode
from aras.scraping.parsers.scholar_parser import parse_google_scholar
from aras.scraping.cache import RedisCache


class ScrapingAgent(BaseAgent):
    """Scrape research sources and emit structured items."""

    def __init__(
        self,
        settings: Settings,
        memory: MemoryAgent,
        on_event: EventSink,
        on_tokens=None,
        on_chat_result=None,
    ) -> None:
        super().__init__(agent_id="scraping", on_event=on_event, on_tokens=on_tokens, on_chat_result=on_chat_result)
        self.settings = settings
        self.memory = memory
        self.router = ScrapeRouter(settings=settings)
        self._cache = RedisCache(settings=settings)

    async def scrape(self, *, plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Scrape a set of sources for the plan keywords."""
        keywords = plan.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        q = " ".join([str(k) for k in keywords if k][:6]).strip() or "machine learning"
        self.emit(f"Scraping query='{q}'")

        urls: list[str] = []
        urls += await self._arxiv_urls(query=q)
        urls += self._wikipedia_urls(query=q)
        urls += self._github_search_urls(query=q)
        urls += self._paperswithcode_urls(query=q)
        urls += self._google_scholar_urls(query=q)

        # Dedup and cap for HTML crawling.
        seen: set[str] = set()
        uniq: list[str] = []
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            uniq.append(u)
        uniq = uniq[:18]

        # Tier 1–2: router (cache → Playwright/js → Scrapling → httpx).
        pages = await asyncio.gather(*[self.router.fetch(u) for u in uniq], return_exceptions=True)
        items: list[dict[str, Any]] = []
        for p in pages:
            if isinstance(p, Exception):
                self.emit(f"Fetch failed: {p}", level="error")
                continue
            url = p.url
            html = p.html
            try:
                if "arxiv.org/abs/" in url:
                    items.append(parse_arxiv(url, html).to_dict())
                elif "github.com/" in url and url.count("/") >= 4:
                    items.append(parse_github(url, html).to_dict())
                elif "paperswithcode.com" in url:
                    items.append(parse_paperswithcode(url, html).to_dict())
                elif "scholar.google.com" in url:
                    items.append(parse_google_scholar(url, html).to_dict())
                else:
                    src = "wikipedia" if "wikipedia.org" in url else "generic"
                    items.append(parse_generic(url, html, source=src).to_dict())
            except Exception as e:
                self.emit(f"Parse failed {url}: {e}", level="error")

        # Tier 3: deep API sources (Semantic Scholar + PapersWithCode API).
        api_items_ss = await self._semantic_scholar_items(query=q, keywords=keywords)
        api_items_pwc = await self._paperswithcode_api_items(query=q, keywords=keywords)
        items.extend(api_items_ss)
        items.extend(api_items_pwc)

        # Rank loosely by relevance + keyword overlap.
        kws = {str(k).lower() for k in keywords if k}
        for it in items:
            txt = (it.get("title", "") + " " + it.get("abstract", "")).lower()
            overlap = sum(1 for k in kws if k and k in txt)
            base_rel = float(it.get("relevance", 0.4))
            # Give a small boost to high-signal scientific sources.
            if it.get("source") in {"arxiv", "semantic_scholar", "paperswithcode", "google_scholar"}:
                base_rel += 0.1
            it["relevance"] = base_rel + min(0.5, 0.05 * overlap)
        items.sort(key=lambda x: float(x.get("relevance", 0.0)), reverse=True)
        self.emit(f"Scraped {len(items)} parsed items (HTML + API)")
        return items

    async def _arxiv_urls(self, *, query: str) -> list[str]:
        api = f"https://export.arxiv.org/api/query?search_query=all:{quote_plus(query)}&start=0&max_results=8"
        async with httpx.AsyncClient(timeout=40) as client:
            r = await client.get(api, headers={"User-Agent": "ARAS/1.0"})
            r.raise_for_status()
            xml = r.text
        soup = BeautifulSoup(xml, "xml")
        urls: list[str] = []
        for entry in soup.find_all("entry"):
            id_el = entry.find("id")
            if not id_el:
                continue
            u = id_el.get_text(strip=True)
            if u:
                urls.append(u.replace("http://", "https://"))
        self.emit(f"arXiv: found {len(urls)} entries")
        return urls

    def _wikipedia_urls(self, *, query: str) -> list[str]:
        # Use the Wikipedia search endpoint (static HTML).
        return [f"https://en.wikipedia.org/w/index.php?search={quote_plus(query)}"]

    def _github_search_urls(self, *, query: str) -> list[str]:
        # GitHub search results are JS-heavy; we still fetch static HTML for a few repo pages by heuristics.
        # We fetch the search page and parse repo links later by generic parser (signal).
        return [f"https://github.com/search?q={quote_plus(query)}&type=repositories"]

    def _paperswithcode_urls(self, *, query: str) -> list[str]:
        return [f"https://paperswithcode.com/search?q={quote_plus(query)}"]

    def _google_scholar_urls(self, *, query: str) -> list[str]:
        # Best-effort: Google Scholar may block automated traffic; router will attempt Playwright and fall back.
        return [f"https://scholar.google.com/scholar?q={quote_plus(query)}"]

    async def _semantic_scholar_items(self, *, query: str, keywords: list[Any]) -> list[dict[str, Any]]:
        """Query Semantic Scholar API for additional high-signal papers."""
        api_q = query or "machine learning"
        cache_key = f"ss:search:{api_q}"
        cached = self._cache.get(cache_key)
        if cached:
            try:
                import json

                data = json.loads(cached)
                return data if isinstance(data, list) else []
            except Exception:
                pass

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": api_q,
            "limit": 10,
            "fields": "title,abstract,year,authors,url,citationCount,isOpenAccess",
        }
        try:
            async with httpx.AsyncClient(timeout=40) as client:
                r = await client.get(url, params=params, headers={"User-Agent": "ARAS/1.0"})
                r.raise_for_status()
                payload = r.json()
        except Exception as e:
            self.emit(f"Semantic Scholar API failed: {e}", level="error")
            return []

        results = payload.get("data") or []
        out: list[dict[str, Any]] = []
        kws = {str(k).lower() for k in keywords if k}
        for rec in results:
            title = str(rec.get("title") or "").strip()
            abstract = str(rec.get("abstract") or "").strip()
            authors = [a.get("name", "") for a in rec.get("authors") or [] if a.get("name")]
            year = rec.get("year")
            url_p = rec.get("url") or ""
            base_rel = 0.8
            txt = (title + " " + abstract).lower()
            overlap = sum(1 for k in kws if k and k in txt)
            rel = base_rel + min(0.5, 0.05 * overlap)
            out.append(
                {
                    "source": "semantic_scholar",
                    "url": url_p or "",
                    "title": (title or url_p)[:300],
                    "abstract": abstract[:4000],
                    "authors": authors[:30],
                    "published": str(year) if year else None,
                    "relevance": float(rel),
                    "extra": {
                        "citation_count": rec.get("citationCount"),
                        "is_open_access": rec.get("isOpenAccess"),
                        "paperId": rec.get("paperId"),
                    },
                }
            )

        try:
            import json

            self._cache.set(cache_key, json.dumps(out))
        except Exception:
            pass
        return out

    async def _paperswithcode_api_items(self, *, query: str, keywords: list[Any]) -> list[dict[str, Any]]:
        """Query PapersWithCode API for tasks/papers."""
        api_q = query or "machine learning"
        cache_key = f"pwc:search:{api_q}"
        cached = self._cache.get(cache_key)
        if cached:
            try:
                import json

                data = json.loads(cached)
                return data if isinstance(data, list) else []
            except Exception:
                pass

        # Use the search API for papers.
        url = "https://paperswithcode.com/api/v1/search/"
        params = {"q": api_q, "page": 1}
        try:
            async with httpx.AsyncClient(timeout=40) as client:
                r = await client.get(url, params=params, headers={"User-Agent": "ARAS/1.0"})
                r.raise_for_status()
                payload = r.json()
        except Exception as e:
            self.emit(f"PapersWithCode API failed: {e}", level="error")
            return []

        results = payload.get("results") or []
        out: list[dict[str, Any]] = []
        kws = {str(k).lower() for k in keywords if k}
        for rec in results[:15]:
            paper = rec.get("paper") or {}
            repo = rec.get("repository") or {}
            title = str(paper.get("title") or "").strip()
            abstract = str(paper.get("abstract") or "").strip()
            url_p = paper.get("url") or rec.get("url") or ""
            authors = []
            base_rel = 0.8
            txt = (title + " " + abstract).lower()
            overlap = sum(1 for k in kws if k and k in txt)
            rel = base_rel + min(0.5, 0.05 * overlap)
            out.append(
                {
                    "source": "paperswithcode_api",
                    "url": url_p,
                    "title": (title or url_p)[:300],
                    "abstract": abstract[:4000],
                    "authors": authors,
                    "published": paper.get("published") or None,
                    "relevance": float(rel),
                    "extra": {
                        "tasks": rec.get("tasks"),
                        "methods": rec.get("methods"),
                        "repository_url": repo.get("url"),
                    },
                }
            )

        try:
            import json

            self._cache.set(cache_key, json.dumps(out))
        except Exception:
            pass
        return out

