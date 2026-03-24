from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

from aras.scraping.parsers.generic_parser import ParsedItem


def parse_arxiv(url: str, html: str, *, relevance: float = 0.8) -> ParsedItem:
    """Parse arXiv abstract pages."""
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one("h1.title")
    title = title_el.get_text(" ", strip=True).replace("Title:", "").strip() if title_el else ""
    abs_el = soup.select_one("blockquote.abstract")
    abstract = abs_el.get_text(" ", strip=True).replace("Abstract:", "").strip() if abs_el else ""
    authors = [a.get_text(strip=True) for a in soup.select("div.authors a")]
    date_el = soup.select_one("div.dateline")
    published = date_el.get_text(" ", strip=True) if date_el else None
    return ParsedItem(
        source="arxiv",
        url=url,
        title=title[:300] if title else url,
        abstract=abstract[:4000],
        authors=authors[:30],
        published=published,
        relevance=float(relevance),
        extra={},
    )

