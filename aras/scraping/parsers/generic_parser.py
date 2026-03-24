from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup


@dataclass
class ParsedItem:
    source: str
    url: str
    title: str
    abstract: str
    authors: list[str]
    published: str | None
    relevance: float
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "published": self.published,
            "relevance": self.relevance,
            "extra": self.extra,
        }


def parse_generic(url: str, html: str, *, source: str = "generic", relevance: float = 0.4) -> ParsedItem:
    """Heuristic extraction for title and summary-like content."""
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else "").strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else url

    # Try meta description
    abstract = ""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        abstract = str(meta.get("content")).strip()
    if not abstract:
        p = soup.find("p")
        abstract = p.get_text(" ", strip=True) if p else ""
    abstract = abstract[:2500]

    published = datetime.now(timezone.utc).date().isoformat()
    return ParsedItem(
        source=source,
        url=url,
        title=title[:300],
        abstract=abstract,
        authors=[],
        published=published,
        relevance=float(relevance),
        extra={},
    )

