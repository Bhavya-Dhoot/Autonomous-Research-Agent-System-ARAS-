from __future__ import annotations

import re

from bs4 import BeautifulSoup

from aras.scraping.parsers.generic_parser import ParsedItem


def parse_google_scholar(url: str, html: str, *, relevance: float = 0.7) -> ParsedItem:
    """Parse Google Scholar result pages (best-effort; may be blocked)."""
    soup = BeautifulSoup(html, "lxml")

    # Scholar result titles are usually in h3.gs_rt
    first = soup.select_one("div.gs_r")
    title = ""
    abstract = ""
    if first:
        t = first.select_one("h3.gs_rt")
        if t:
            title = t.get_text(" ", strip=True)
        a = first.select_one("div.gs_rs")
        if a:
            abstract = a.get_text(" ", strip=True)
    if not title:
        title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip() or url
    if not abstract:
        abstract = ""

    # Extract year if present in the meta line
    published = None
    meta = first.select_one("div.gs_a").get_text(" ", strip=True) if first and first.select_one("div.gs_a") else ""
    m = re.search(r"(19|20)\d{2}", meta)
    if m:
        published = m.group(0)

    return ParsedItem(
        source="google_scholar",
        url=url,
        title=title[:300],
        abstract=abstract[:2500],
        authors=[],
        published=published,
        relevance=float(relevance),
        extra={"meta": meta[:500]},
    )

