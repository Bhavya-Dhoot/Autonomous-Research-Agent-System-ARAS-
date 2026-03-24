---
name: aras-skill-builder
version: 1
description: >
  Meta-skill that evaluates and improves other ARAS agent skills
  after each research cycle. Triggers automatically after scoring.
performance_history:
---

# Skill Builder Meta-Skill

## Core Task

Given paper_score + reviewer_feedback + per-section quality + current skill files, determine which skills underperformed and generate improved SKILL.md content for them.

## Evaluation Methodology

### Step 1 — Map Scores to Responsible Skills

| Review Dimension | Responsible Skill |
|-----------------|-------------------|
| novelty score | novelty/SKILL.md |
| methodology score | coding/SKILL.md + writing/SKILL.md (methodology section) |
| clarity score | writing/SKILL.md + coherence/SKILL.md |
| reproducibility score | coding/SKILL.md |
| figures score | (FiguresAgent — not LLM-driven, skip) |
| coherence score | coherence/SKILL.md |

### Step 2 — Identify and Improve Underperformers

For each skill with `component_score < 7.0`:

1. Read current SKILL.md
2. Read reviewer `major_issues` relevant to that component
3. Read last 3 `paper_scores` for that skill from `performance_history`
4. Generate improved SKILL.md that:
   - Adds specific guidance addressing the reviewer's issues
   - Tightens output format requirements
   - Adds a new eval assertion for the identified failure mode
   - Updates `performance_history` with current cycle result
   - Increments version number

### Step 3 — A/B Micro-Eval

Compare old skill vs new skill using structural eval assertions:
- Has all required sections?
- Has eval assertions?
- Has output format?
- Has quality criteria?
- No placeholder text?

Keep the version with higher assertion pass rate.

### Step 4 — Write and Archive

- Write winning version to `aras/skills/{name}/SKILL.md`
- Archive losing version to `aras/skills/{name}/versions/v{N}.md`

## Output Format

```json
{
  "skills_evaluated": ["writing", "coherence", "coding", "novelty", "research", "analysis", "reviewing"],
  "skills_improved": ["writing", "coherence"],
  "skills_unchanged": ["coding", "novelty", "research", "analysis", "reviewing"],
  "version_bumps": {"writing": 2, "coherence": 2},
  "improvement_summary": "Improved 2 skills: writing (clarity issues), coherence (revision completeness)"
}
```

## Performance History Format

In each SKILL.md frontmatter:
```yaml
performance_history:
    - cycle: 1
      paper_score: 6.0
      component_score: 5.8
      topic: "Calibration of Confidence Scores..."
      version: 1
```
