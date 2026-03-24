"""Tests for the SkillLoader and SkillBuilderAgent."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from aras.skills.skill_loader import SkillLoader


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """Create a minimal skills directory with a test skill."""
    skill_dir = tmp_path / "research"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        dedent("""\
            ---
            name: test-research-planner
            version: 3
            description: Test skill for research planning.
            performance_history:
            ---

            # Research Planner Skill

            ## Core Task

            Generate research plans in JSON format.

            ## Output Format

            Return JSON with keys: hypothesis, questions, experiments.

            ## Quality Criteria

            - Hypothesis must be falsifiable

            ## Eval Assertions

            - **json_parseable**: output is valid JSON
            - **has_all_keys**: all required keys present
        """),
        encoding="utf-8",
    )
    return tmp_path


class TestSkillLoader:
    def test_load_strips_frontmatter(self, skills_dir: Path) -> None:
        loader = SkillLoader(skills_dir=skills_dir)
        content = loader.load("research")
        # YAML frontmatter keys should not appear in loaded content
        assert "name: test-research-planner" not in content
        assert "version: 3" not in content
        assert "# Research Planner Skill" in content

    def test_load_caches(self, skills_dir: Path) -> None:
        loader = SkillLoader(skills_dir=skills_dir)
        first = loader.load("research")
        second = loader.load("research")
        assert first is second  # same object from cache

    def test_load_missing_skill_returns_empty(self, skills_dir: Path) -> None:
        loader = SkillLoader(skills_dir=skills_dir)
        assert loader.load("nonexistent_agent") == ""

    def test_get_version(self, skills_dir: Path) -> None:
        loader = SkillLoader(skills_dir=skills_dir)
        assert loader.get_version("research") == 3

    def test_get_version_missing(self, skills_dir: Path) -> None:
        loader = SkillLoader(skills_dir=skills_dir)
        assert loader.get_version("nonexistent") == 0

    def test_get_frontmatter(self, skills_dir: Path) -> None:
        loader = SkillLoader(skills_dir=skills_dir)
        fm = loader.get_frontmatter("research")
        assert fm["name"] == "test-research-planner"
        assert fm["version"] == "3"

    def test_get_eval_assertions(self, skills_dir: Path) -> None:
        loader = SkillLoader(skills_dir=skills_dir)
        assertions = loader.get_eval_assertions("research")
        assert "json_parseable" in assertions
        assert "has_all_keys" in assertions

    def test_invalidate_cache(self, skills_dir: Path) -> None:
        loader = SkillLoader(skills_dir=skills_dir)
        first = loader.load("research")
        loader.invalidate_cache("research")
        second = loader.load("research")
        assert first is not second  # different objects after invalidation
        assert first == second  # same content though

    def test_invalidate_cache_all(self, skills_dir: Path) -> None:
        loader = SkillLoader(skills_dir=skills_dir)
        loader.load("research")
        assert len(loader._cache) == 1
        loader.invalidate_cache()
        assert len(loader._cache) == 0

    def test_build_system_prompt_with_skill(self, skills_dir: Path) -> None:
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.build_system_prompt("research", "You are a helper.")
        assert "# SKILL INSTRUCTIONS" in result
        assert "# Research Planner Skill" in result
        assert "# ADDITIONAL CONTEXT" in result
        assert "You are a helper." in result

    def test_build_system_prompt_without_skill(self, skills_dir: Path) -> None:
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.build_system_prompt("nonexistent", "You are a helper.")
        assert result == "You are a helper."

    def test_build_system_prompt_no_skill_file(self, tmp_path: Path) -> None:
        """If skills_dir is empty, returns base prompt unchanged."""
        loader = SkillLoader(skills_dir=tmp_path)
        result = loader.build_system_prompt("research", "Base prompt")
        assert result == "Base prompt"


class TestSkillLoaderAllSkills:
    """Verify all production SKILL.md files load correctly."""

    @pytest.fixture
    def production_loader(self) -> SkillLoader:
        skills_dir = Path(__file__).resolve().parents[2] / "aras" / "skills"
        return SkillLoader(skills_dir=skills_dir)

    @pytest.mark.parametrize(
        "agent_name",
        ["research", "novelty", "writer", "reviewer", "coherence", "analyst", "coder", "skill_builder"],
    )
    def test_production_skill_loads(self, production_loader: SkillLoader, agent_name: str) -> None:
        content = production_loader.load(agent_name)
        assert content, f"Skill for {agent_name} should not be empty"
        # Frontmatter YAML keys should be stripped (name:, version:, description:)
        assert "name:" not in content.split("\n")[0], f"Frontmatter keys should be stripped for {agent_name}"

    @pytest.mark.parametrize(
        "agent_name",
        ["research", "novelty", "writer", "reviewer", "coherence", "analyst", "coder", "skill_builder"],
    )
    def test_production_skill_has_version(self, production_loader: SkillLoader, agent_name: str) -> None:
        version = production_loader.get_version(agent_name)
        assert version >= 1, f"Skill {agent_name} should have version >= 1"

    @pytest.mark.parametrize(
        "agent_name",
        ["research", "novelty", "writer", "reviewer", "coherence", "analyst", "coder"],
    )
    def test_production_skill_has_eval_assertions(self, production_loader: SkillLoader, agent_name: str) -> None:
        assertions = production_loader.get_eval_assertions(agent_name)
        assert len(assertions) >= 2, f"Skill {agent_name} should have at least 2 eval assertions"
