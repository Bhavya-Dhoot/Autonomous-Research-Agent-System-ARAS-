from __future__ import annotations

from bs4 import BeautifulSoup

from aras.scraping.parsers.generic_parser import ParsedItem


def parse_paperswithcode(url: str, html: str, *, relevance: float = 0.75) -> ParsedItem:
    """Parse PapersWithCode search/task pages for high-signal snippets."""
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else url

    # PWC often has short card summaries; take the first few.
    cards = soup.select(".paper-card, .task-content, .content")
    abstract = ""
    for c in cards[:4]:
        t = c.get_text(" ", strip=True)
        if t and len(t) > len(abstract):
            abstract = t
    if not abstract:
        p = soup.find("p")
        abstract = p.get_text(" ", strip=True) if p else ""
    abstract = abstract[:3500]

    return ParsedItem(
        source="paperswithcode",
        url=url,
        title=title[:300],
        abstract=abstract,
        authors=[],
        published=None,
        relevance=float(relevance),
        extra={},
    )

