from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class BibEntry:
    key: str
    title: str
    author: str
    year: str
    url: str
    note: str = ""

    def to_bibtex(self) -> str:
        t = _escape(self.title)
        a = _escape(self.author)
        y = _escape(self.year)
        u = _escape(self.url)
        n = _escape(self.note)
        return (
            f"@misc{{{self.key},\n"
            f"  title={{ {t} }},\n"
            f"  author={{ {a} }},\n"
            f"  year={{ {y} }},\n"
            f"  howpublished={{\\url{{{u}}}}},\n"
            f"  note={{ {n} }}\n"
            f"}}\n"
        )


def build_bibtex(scraped: list[dict[str, Any]], *, max_entries: int = 25) -> tuple[str, list[BibEntry]]:
    """Create BibTeX from scraped items; return bibtex string and entries."""
    entries: list[BibEntry] = []
    seen: set[str] = set()
    for it in scraped:
        title = str(it.get("title") or "").strip()
        url = str(it.get("url") or "").strip()
        if not title or not url:
            continue
        key = _key_from(title)
        if key in seen:
            continue
        seen.add(key)
        authors = it.get("authors") or []
        author = ", ".join([str(a) for a in authors[:6]]) if authors else "Unknown"
        year = _year_from(str(it.get("published") or "")) or "2026"
        note = f"source={it.get('source','')}"
        entries.append(BibEntry(key=key, title=title, author=author, year=year, url=url, note=note))
        if len(entries) >= max_entries:
            break
    bib = "\n".join([e.to_bibtex() for e in entries])
    return bib, entries


def cite_keys(entries: list[BibEntry], *, n: int = 15) -> str:
    ks = [e.key for e in entries[:n]]
    return ", ".join([f"\\cite{{{k}}}" for k in ks]) if ks else ""


def _key_from(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:50] or "ref")


def _year_from(s: str) -> str | None:
    m = re.search(r"(19|20)\\d{2}", s)
    return m.group(0) if m else None


def _escape(s: str) -> str:
    return s.replace("{", "\\{").replace("}", "\\}").replace("&", "\\&").replace("%", "\\%")

