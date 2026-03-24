from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Template

from aras.paper.bibliography import build_bibtex, cite_keys
from aras.utils.fs import safe_write_text
from aras.utils.logging import get_logger


log = get_logger("latex")


@dataclass
class PaperArtifacts:
    tex: Path
    pdf: Path | None
    bib: Path


class LaTeXPaperBuilder:
    """Build and compile LaTeX paper from structured section texts."""

    def __init__(self, template_path: Path) -> None:
        self.template_path = template_path

    def render(
        self,
        *,
        title: str,
        abstract: str,
        keywords: str,
        sections: dict[str, str],
        scraped: list[dict[str, Any]],
        out_dir: Path,
    ) -> PaperArtifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
        tpl = Template(self.template_path.read_text(encoding="utf-8"))

        bibtex, entries = build_bibtex(scraped, max_entries=30)
        bib_path = out_dir / "references.bib"
        safe_write_text(bib_path, bibtex)

        cites = cite_keys(entries, n=15)
        related_work = sections.get("related_work", "")
        if cites and cites not in related_work:
            related_work = related_work + "\n\nKey references: " + cites

        tex = tpl.render(
            title=title,
            abstract=abstract,
            keywords=keywords,
            introduction=sections.get("introduction", ""),
            related_work=related_work,
            methodology=sections.get("methodology", ""),
            architecture=sections.get("architecture", ""),
            experiments=sections.get("experiments", ""),
            results=sections.get("results", ""),
            discussion=sections.get("discussion", ""),
            conclusion=sections.get("conclusion", ""),
        )

        tex_path = out_dir / "paper.tex"
        safe_write_text(tex_path, tex)
        return PaperArtifacts(tex=tex_path, pdf=None, bib=bib_path)

    async def compile(self, *, out_dir: Path, tex_path: Path) -> Path | None:
        """Compile to PDF using tectonic or pdflatex if available."""
        pdf_path = out_dir / "paper.pdf"

        engine = shutil.which("tectonic") or shutil.which("pdflatex")
        if not engine:
            log.warning("No TeX engine found (tectonic/pdflatex). Skipping PDF compile.")
            return None

        if "tectonic" in Path(engine).name.lower():
            cmd = [engine, "--outdir", str(out_dir), str(tex_path)]
        else:
            cmd = [engine, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(out_dir), str(tex_path)]

        log.info("Compiling paper: %s", " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(out_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_b, err_b = await proc.communicate()
        if proc.returncode != 0:
            log.error("TeX compile failed: %s", (err_b.decode("utf-8", "replace") + out_b.decode("utf-8", "replace"))[-8000:])
            return None
        return pdf_path if pdf_path.exists() else None

