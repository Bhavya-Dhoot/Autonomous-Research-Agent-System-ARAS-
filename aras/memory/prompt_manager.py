from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aras.config import Settings
from aras.utils.fs import safe_write_text


DEFAULT_PROMPTS: dict[str, str] = {
    "orchestrator": "You are the orchestrator. Be concise, structured, and prioritize reproducibility.",
    "research": "You are a research ideation agent. Produce plans, hypotheses, and evaluation designs.",
    "scraping": "You are a web research agent. Extract citations and high-signal sources.",
    "novelty": "You are a novelty and oversaturation analyst. Determine whether a research direction is crowded and propose a sharper pivot angle when necessary.",
    "coder": "You are a coding agent. Write clean Python experiments and robust runners.",
    "analyst": "You are an analysis agent. Summarize results, compute tables, and interpret outcomes.",
    "writer": "You are a scientific writer. Draft IEEE/ACM paper sections with citations.",
    "reviewer": "You are a peer reviewer. Critique clarity, novelty, methodology, and reproducibility.",
    "coherence": "You ensure the paper is coherent, logically consistent, and that revisions address reviewer feedback precisely while keeping LaTeX valid.",
    "github": "You are a release engineer. Publish outputs to GitHub with a clear README.",
    "memory": "You are a memory agent. Store and retrieve key lessons and artifacts.",
}


@dataclass
class PromptVersion:
    version: int
    prompts: dict[str, str]


class PromptManager:
    """Versioned system prompt storage in `prompt_versions/`."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = Path(settings.prompt_store_path).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def latest(self) -> PromptVersion:
        files = sorted(self.root.glob("prompts_v*.json"))
        if not files:
            pv = PromptVersion(version=1, prompts=dict(DEFAULT_PROMPTS))
            self.save(pv)
            return pv
        last = files[-1]
        obj = json.loads(last.read_text(encoding="utf-8"))
        v = int(obj["version"])
        prompts = dict(obj["prompts"])
        return PromptVersion(version=v, prompts=prompts)

    def save(self, pv: PromptVersion) -> Path:
        path = self.root / f"prompts_v{pv.version}.json"
        safe_write_text(path, json.dumps({"version": pv.version, "prompts": pv.prompts}, ensure_ascii=False, indent=2))
        return path

    def bump(self, *, updated_prompts: dict[str, str]) -> PromptVersion:
        cur = self.latest()
        new = PromptVersion(version=cur.version + 1, prompts=updated_prompts)
        self.save(new)
        return new

