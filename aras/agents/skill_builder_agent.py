from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from aras.agents.base import BaseAgent, EventSink
from aras.agents.memory_agent import MemoryAgent
from aras.config import Settings
from aras.healing.fallback_router import FallbackRouter
from aras.skills.skill_loader import SkillLoader
from aras.utils.logging import get_logger

log = get_logger("skill-builder")


class SkillBuilderAgent(BaseAgent):
    """Meta-agent that improves ARAS skills after each research cycle.

    Uses the skill-creator methodology: evaluate → identify failures →
    generate improved skill → run micro-eval → keep winner.
    """

    def __init__(
        self,
        settings: Settings,
        memory: MemoryAgent,
        skill_loader: SkillLoader,
        on_event: EventSink,
        on_tokens=None,
        on_chat_result=None,
    ) -> None:
        super().__init__(
            agent_id="skill_builder",
            on_event=on_event,
            on_tokens=on_tokens,
            on_chat_result=on_chat_result,
        )
        self.settings = settings
        self.memory = memory
        self.skill_loader = skill_loader
        self.router = FallbackRouter(settings=settings)

    async def improve_skills(
        self,
        *,
        cycle: int,
        topic: str,
        paper_score: float,
        reviewer_feedback: dict[str, Any],
        section_scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Main entry point. Called by orchestrator after scoring.

        Returns improvement summary dict.
        """
        self.emit(f"Skill builder starting. Paper score: {paper_score:.1f}")

        # 1. Map scores to responsible skills
        skill_scores = self._map_scores_to_skills(
            paper_score=paper_score,
            reviewer_feedback=reviewer_feedback,
            section_scores=section_scores or {},
        )

        # 2. Identify underperforming skills (score < 7.0)
        underperforming = {
            name: score for name, score in skill_scores.items() if score < 7.0
        }

        self.emit(f"Skills to improve: {list(underperforming.keys())}")

        improved: list[str] = []
        unchanged: list[str] = []
        version_bumps: dict[str, int] = {}

        for skill_name, score in underperforming.items():
            success = await self._improve_skill(
                skill_name=skill_name,
                component_score=score,
                cycle=cycle,
                topic=topic,
                paper_score=paper_score,
                reviewer_feedback=reviewer_feedback,
            )
            if success:
                improved.append(skill_name)
                version_bumps[skill_name] = self.skill_loader.get_version(skill_name)
            else:
                unchanged.append(skill_name)

        # 3. Update performance history for all skills
        for skill_name, score in skill_scores.items():
            if skill_name not in underperforming:
                unchanged.append(skill_name)
            self._update_performance_history(
                skill_name, cycle, topic, paper_score, score
            )

        summary = self._build_summary(improved, unchanged, version_bumps)
        self.emit(f"Skill builder done. {summary}")

        return {
            "skills_evaluated": list(skill_scores.keys()),
            "skills_improved": improved,
            "skills_unchanged": unchanged,
            "version_bumps": version_bumps,
            "improvement_summary": summary,
        }

    async def _improve_skill(
        self,
        *,
        skill_name: str,
        component_score: float,
        cycle: int,
        topic: str,
        paper_score: float,
        reviewer_feedback: dict[str, Any],
    ) -> bool:
        """Generate improved SKILL.md and validate it passes more eval assertions."""
        current_skill = self._read_skill_file(skill_name)
        if not current_skill:
            self.emit(f"Skill file not found for {skill_name} — skipping", level="error")
            return False

        # Extract relevant reviewer issues for this skill
        relevant_issues = self._extract_relevant_issues(skill_name, reviewer_feedback)

        # Generate improved skill via LLM
        prompt = self._build_improvement_prompt(
            skill_name=skill_name,
            current_skill=current_skill,
            component_score=component_score,
            relevant_issues=relevant_issues,
            cycle=cycle,
            topic=topic,
        )

        try:
            rag = self.memory.rag_context(query=f"skill improvement for {skill_name}")
            system = (
                "You are improving an ARAS agent skill file. "
                "Output ONLY the complete new SKILL.md content. "
                "Start with --- YAML frontmatter. No explanation.\n\n"
                f"RAG CONTEXT:\n{rag}"
            )
            res = await self.router.chat(
                role_system=system,
                messages=[{"role": "user", "content": prompt}],
                purpose=f"skill_improvement_{skill_name}",
                prefer=["nvidia", "openai", "local"],
                thinking=True,
                temperature=0.4,
                max_tokens=16383,
            )
            self.record_chat_result(res)
            try:
                self.add_tokens(res.tokens_used)
            except Exception:
                pass
        except Exception as e:
            self.emit(f"LLM failed for skill {skill_name}: {e}", level="error")
            return False

        new_skill_content = (res.text or "").strip()
        if not new_skill_content or "---" not in new_skill_content:
            self.emit(f"Skill {skill_name}: LLM returned invalid content — keeping current")
            return False

        # Validate new skill passes more eval assertions than old
        old_score = self._run_micro_eval(current_skill)
        new_score = self._run_micro_eval(new_skill_content)

        if new_score >= old_score:
            # New version wins — save it
            self._save_skill_version(skill_name, current_skill)  # archive old
            self._write_skill_file(skill_name, new_skill_content)
            self.skill_loader.invalidate_cache(skill_name)
            self.emit(
                f"Skill {skill_name} improved: "
                f"eval {old_score:.0%} → {new_score:.0%}, version bumped"
            )
            return True
        else:
            self.emit(
                f"Skill {skill_name}: new version scored {new_score:.0%} vs "
                f"current {old_score:.0%} — keeping current"
            )
            return False

    def _run_micro_eval(self, skill_content: str) -> float:
        """Run structural eval assertions on skill content.

        Returns fraction of assertions that pass (0.0 to 1.0).
        Checks structural properties of the skill content itself.
        """
        assertions_passed = 0
        total_assertions = 6

        # 1. Has eval assertions section
        if "Eval" in skill_content and ("assertion" in skill_content.lower()):
            assertions_passed += 1

        # 2. Has output format section
        if "Output" in skill_content and ("JSON" in skill_content or "LaTeX" in skill_content or "json" in skill_content):
            assertions_passed += 1

        # 3. Has quality criteria
        if "Quality" in skill_content or "quality" in skill_content:
            assertions_passed += 1

        # 4. No placeholder text
        if "TODO" not in skill_content and "PLACEHOLDER" not in skill_content:
            assertions_passed += 1

        # 5. Has version in frontmatter
        if "version:" in skill_content:
            assertions_passed += 1

        # 6. Has YAML frontmatter
        if skill_content.strip().startswith("---"):
            assertions_passed += 1

        return assertions_passed / total_assertions

    def _build_improvement_prompt(
        self,
        *,
        skill_name: str,
        current_skill: str,
        component_score: float,
        relevant_issues: list[str],
        cycle: int,
        topic: str,
    ) -> str:
        issues_text = "\n".join(f"  - {issue}" for issue in relevant_issues) if relevant_issues else "  (no specific issues identified)"
        return (
            f"You are improving an ARAS agent skill file.\n\n"
            f"SKILL NAME: {skill_name}\n"
            f"CURRENT VERSION:\n{current_skill}\n\n"
            f"PERFORMANCE THIS CYCLE:\n"
            f"- Topic: {topic}\n"
            f"- Component score: {component_score:.1f}/10\n"
            f"- Cycle: {cycle}\n"
            f"- Relevant reviewer issues:\n{issues_text}\n\n"
            f"YOUR TASK:\n"
            f"Rewrite the SKILL.md to address these specific issues.\n"
            f"The improved skill must:\n"
            f"1. Add specific guidance that would have prevented the identified issues\n"
            f"2. Tighten output format requirements to reduce LLM ambiguity\n"
            f"3. Add at least one new eval assertion targeting the failure mode\n"
            f"4. Update performance_history in the frontmatter with this cycle's result\n"
            f"5. Increment the version number by 1\n"
            f"6. Keep everything that was working well\n\n"
            f"Output ONLY the complete new SKILL.md content.\n"
            f"Start with the --- YAML frontmatter block.\n"
            f"No explanation. No markdown code fences. Just the raw SKILL.md content."
        )

    def _map_scores_to_skills(
        self,
        paper_score: float,
        reviewer_feedback: dict[str, Any],
        section_scores: dict[str, float],
    ) -> dict[str, float]:
        """Map reviewer dimension scores to responsible skill names."""
        scores = reviewer_feedback.get("scores", {})
        if not isinstance(scores, dict):
            scores = {}
        return {
            "novelty": float(scores.get("novelty", paper_score)),
            "writing": float(scores.get("clarity", paper_score)),
            "reviewing": paper_score,
            "coherence": float(scores.get("coherence", paper_score)),
            "coding": float(scores.get("reproducibility", paper_score)),
            "analysis": float(scores.get("methodology", paper_score)),
            "research": paper_score,
        }

    def _extract_relevant_issues(
        self, skill_name: str, reviewer_feedback: dict[str, Any]
    ) -> list[str]:
        """Pull major_issues from reviewer feedback relevant to this skill."""
        skill_to_keywords: dict[str, list[str]] = {
            "writing": ["introduction", "related", "methodology", "results", "conclusion", "clarity", "writing"],
            "coherence": ["coherence", "flow", "transition", "consistency", "revision"],
            "reviewing": [],
            "coding": ["experiment", "reproducib", "dataset", "baseline", "code"],
            "analysis": ["results", "analysis", "finding", "metric", "table"],
            "novelty": ["novelty", "contribution", "prior work", "related work"],
            "research": ["hypothesis", "research question", "plan", "design"],
        }

        keywords = skill_to_keywords.get(skill_name, [])
        all_issues: list[str] = []
        for key in ("major_issues", "minor_issues"):
            val = reviewer_feedback.get(key)
            if isinstance(val, list):
                all_issues.extend(str(i) for i in val)

        if not keywords:
            return all_issues[:3]

        relevant = [
            issue
            for issue in all_issues
            if any(kw.lower() in issue.lower() for kw in keywords)
        ]
        return relevant[:5] if relevant else all_issues[:3]

    def _read_skill_file(self, skill_name: str) -> str:
        path = self._skill_path(skill_name)
        if path and path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _write_skill_file(self, skill_name: str, content: str) -> None:
        path = self._skill_path(skill_name)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def _save_skill_version(self, skill_name: str, content: str) -> None:
        """Archive current version before overwriting."""
        path = self._skill_path(skill_name)
        if not path:
            return
        versions_dir = path.parent / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        version = self.skill_loader.get_version(skill_name)
        archive_path = versions_dir / f"v{version}.md"
        archive_path.write_text(content, encoding="utf-8")

    def _update_performance_history(
        self,
        skill_name: str,
        cycle: int,
        topic: str,
        paper_score: float,
        component_score: float,
    ) -> None:
        """Append cycle result to skill frontmatter performance_history."""
        content = self._read_skill_file(skill_name)
        if not content:
            return
        version = self.skill_loader.get_version(skill_name)
        new_entry = (
            f"    - cycle: {cycle}\n"
            f"      paper_score: {paper_score:.1f}\n"
            f"      component_score: {component_score:.1f}\n"
            f'      topic: "{topic[:60]}"\n'
            f"      version: {version}\n"
        )
        # Insert after performance_history: line
        updated = content.replace(
            "performance_history:\n", f"performance_history:\n{new_entry}", 1
        )
        if updated != content:
            self._write_skill_file(skill_name, updated)
            self.skill_loader.invalidate_cache(skill_name)

    def _skill_path(self, skill_name: str) -> Path | None:
        """Resolve the SKILL.md path for a given skill name."""
        # Map skill_builder internal names to SkillLoader agent names
        loader_name_map: dict[str, str] = {
            "writing": "writer",
            "reviewing": "reviewer",
            "analysis": "analyst",
            "coding": "coder",
        }
        loader_key = loader_name_map.get(skill_name, skill_name)
        rel = SkillLoader.SKILL_MAP.get(loader_key, "")
        if not rel:
            rel = f"{skill_name}/SKILL.md"
        return self.skill_loader.skills_dir / rel

    def _build_summary(
        self,
        improved: list[str],
        unchanged: list[str],
        version_bumps: dict[str, int],
    ) -> str:
        lines = [f"Improved {len(improved)} skills: {improved}"]
        for name, ver in version_bumps.items():
            lines.append(f"  {name} → v{ver}")
        if unchanged:
            lines.append(f"Unchanged (score >= 7.0 or improvement failed): {unchanged}")
        return "\n".join(lines)
