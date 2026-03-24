from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from aras.utils.logging import get_logger

log = get_logger("skill-loader")


class SkillLoader:
    """Loads SKILL.md files and injects them into agent system prompts."""

    SKILL_MAP: dict[str, str] = {
        "research": "research/SKILL.md",
        "novelty": "novelty/SKILL.md",
        "writer": "writing/SKILL.md",
        "reviewer": "reviewing/SKILL.md",
        "coherence": "coherence/SKILL.md",
        "analyst": "analysis/SKILL.md",
        "coder": "coding/SKILL.md",
        "skill_builder": "skill-builder/SKILL.md",
    }

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = Path(skills_dir).resolve()
        self._cache: dict[str, str] = {}

    def load(self, agent_name: str) -> str:
        """Load skill content for an agent. Returns skill body (no frontmatter)."""
        if agent_name in self._cache:
            return self._cache[agent_name]
        rel = self.SKILL_MAP.get(agent_name, "")
        if not rel:
            return ""
        path = self.skills_dir / rel
        if not path.exists():
            return ""  # graceful fallback — agent uses its own default prompt
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            log.warning("Failed to read skill file %s: %s", path, e)
            return ""
        # Strip YAML frontmatter (--- block)
        body = re.sub(r"^---.*?---\s*", "", content, count=1, flags=re.DOTALL)
        self._cache[agent_name] = body
        return body

    def get_version(self, agent_name: str) -> int:
        """Read version from SKILL.md frontmatter."""
        rel = self.SKILL_MAP.get(agent_name, "")
        if not rel:
            return 0
        path = self.skills_dir / rel
        if not path.exists():
            return 0
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return 0
        match = re.search(r"^version:\s*(\d+)", content, re.MULTILINE)
        return int(match.group(1)) if match else 1

    def get_frontmatter(self, agent_name: str) -> dict[str, Any]:
        """Parse YAML frontmatter from a SKILL.md. Returns empty dict on failure."""
        rel = self.SKILL_MAP.get(agent_name, "")
        if not rel:
            return {}
        path = self.skills_dir / rel
        if not path.exists():
            return {}
        try:
            content = path.read_text(encoding="utf-8")
            m = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if not m:
                return {}
            # Simple YAML key: value parsing (avoids pyyaml dependency)
            fm: dict[str, Any] = {}
            for line in m.group(1).splitlines():
                line = line.strip()
                if ":" in line and not line.startswith("-") and not line.startswith("#"):
                    key, _, val = line.partition(":")
                    fm[key.strip()] = val.strip()
            return fm
        except Exception:
            return {}

    def get_eval_assertions(self, agent_name: str) -> list[str]:
        """Extract eval assertion names from SKILL.md for SkillBuilderAgent."""
        skill_content = self.load(agent_name)
        # Parse lines under "Eval assertions:" header — format: "- **name**: ..."
        assertions = re.findall(r"-\s+\*\*(\w+)\*\*:", skill_content)
        if not assertions:
            # Fallback: plain format "- name: ..."
            assertions = re.findall(r"-\s+(\w+):", skill_content)
        return assertions

    def invalidate_cache(self, agent_name: str | None = None) -> None:
        """Called after SkillBuilderAgent updates a skill. None = clear all."""
        if agent_name is None:
            self._cache.clear()
        else:
            self._cache.pop(agent_name, None)

    def build_system_prompt(self, agent_name: str, base_prompt: str) -> str:
        """Build a system prompt by prepending skill content to the base prompt.

        If no skill is available, returns the base prompt unchanged.
        This is the main integration point — agents call this instead of
        using the base prompt directly.
        """
        skill_content = self.load(agent_name)
        if not skill_content:
            return base_prompt
        return f"# SKILL INSTRUCTIONS\n{skill_content}\n\n# ADDITIONAL CONTEXT\n{base_prompt}"
