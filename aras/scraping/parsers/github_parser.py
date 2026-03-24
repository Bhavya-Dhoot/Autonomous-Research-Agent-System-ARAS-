from __future__ import annotations

from bs4 import BeautifulSoup

from aras.scraping.parsers.generic_parser import ParsedItem


def parse_github(url: str, html: str, *, relevance: float = 0.7) -> ParsedItem:
    """Parse GitHub repository pages for description + README snippet."""
    soup = BeautifulSoup(html, "lxml")
    title = ""
    h1 = soup.select_one("strong.mr-2.flex-self-stretch a")
    if h1:
        title = h1.get_text(strip=True)
    desc = soup.select_one("p.f4.my-3")
    abstract = desc.get_text(" ", strip=True) if desc else ""
    readme = soup.select_one("article.markdown-body")
    if readme and not abstract:
        abstract = readme.get_text(" ", strip=True)[:2500]
    return ParsedItem(
        source="github",
        url=url,
        title=(title or url)[:300],
        abstract=abstract[:4000],
        authors=[],
        published=None,
        relevance=float(relevance),
        extra={},
    )

