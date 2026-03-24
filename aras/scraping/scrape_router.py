from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from scrapling import Fetcher

from aras.config import Settings
from aras.scraping.cache import RedisCache
from aras.utils.logging import get_logger


log = get_logger("scrape-router")


@dataclass
class ScrapedPage:
    url: str
    html: str
    fetched_via: str

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "fetched_via": self.fetched_via, "html": self.html}


class PlaywrightFetcher:
    """Best-effort async fetcher for JS-heavy pages using Playwright."""

    def __init__(self, *, user_agent: str = "ARAS/1.0") -> None:
        self.user_agent = user_agent

    async def get(self, url: str) -> str:
        """Fetch rendered HTML for a URL."""
        try:
            from playwright.async_api import async_playwright  # type: ignore[import-not-found]
        except Exception as e:
            raise RuntimeError("Playwright not installed; cannot fetch JS-heavy page") from e

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                ctx = await browser.new_context(user_agent=self.user_agent)
                page = await ctx.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                # Best-effort: allow late JS to settle.
                await page.wait_for_timeout(800)
                html = await page.content()
                await ctx.close()
                return html
            finally:
                await browser.close()


class ScrapeRouter:
    """Choose fetch strategy per domain; uses Scrapling Fetcher (static) as default.

    For JS-heavy pages, routes to PlaywrightFetcher when available.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache = RedisCache(settings=settings)
        # Scrapling's API has varied across versions; keep init compatible.
        try:
            self._fetcher = Fetcher(stealthy=True)
        except TypeError:
            self._fetcher = Fetcher()
        self._pw = PlaywrightFetcher(user_agent="ARAS/1.0")

        # Domains/paths that often require JS rendering.
        self._js_heavy_hosts: set[str] = {"github.com", "scholar.google.com", "paperswithcode.com"}

    async def fetch(self, url: str) -> ScrapedPage:
        cached = self.cache.get(url)
        if cached:
            return ScrapedPage(url=url, html=cached, fetched_via="cache")

        host = (urlparse(url).hostname or "").lower()
        if host in self._js_heavy_hosts and ("/search" in url or "scholar?" in url):
            try:
                html = await self._pw.get(url)
                self.cache.set(url, html)
                return ScrapedPage(url=url, html=html, fetched_via="playwright")
            except Exception as e:
                log.warning("Playwright fetch failed (%s), falling back: %s", url, e)

        # Scrapling's Fetcher is sync; run in thread.
        try:
            loop = asyncio.get_running_loop()
            html = await loop.run_in_executor(None, self._fetcher.get, url)
            text = getattr(html, "text", None) or str(html)
            self.cache.set(url, text)
            return ScrapedPage(url=url, html=text, fetched_via="scrapling_fetcher")
        except Exception as e:
            log.warning("Scrapling fetcher failed (%s), falling back to httpx: %s", url, e)
            async with httpx.AsyncClient(timeout=40, follow_redirects=True) as client:
                r = await client.get(url, headers={"User-Agent": "ARAS/1.0"})
                r.raise_for_status()
                text = r.text
            self.cache.set(url, text)
            return ScrapedPage(url=url, html=text, fetched_via="httpx")

