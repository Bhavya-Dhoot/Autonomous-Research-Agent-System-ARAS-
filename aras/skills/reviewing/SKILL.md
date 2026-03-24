---
name: aras-peer-reviewer
version: 1
description: >
  Performs structured peer review of a research paper.
  Triggers when ARAS needs to review paper.tex between revision rounds.
performance_history:
    - cycle: 1
      paper_score: 6.0
      component_score: 6.0
      topic: "Can background music tempo influence decision fatigue? A stu"
      version: 0
    - cycle: 1
      paper_score: 3.0
      component_score: 3.0
      topic: "Uncertainty Quantification in Deep Neural Networks via Ensem"
      version: 0
---

# Peer Reviewer Skill

## Core Task

Given paper.tex (truncated to 16K chars), produce a structured review as **STRICT JSON**.

## Output Format

```json
{
  "overall_score": 7.0,
  "scores": {
    "novelty": 6.5,
    "methodology": 7.0,
    "clarity": 8.0,
    "reproducibility": 6.0,
    "figures": 5.0,
    "coherence": 7.5
  },
  "major_issues": [
    "Section Experiments: No comparison against state-of-the-art baselines. Suggested fix: add XGBoost and Random Forest baselines.",
    "Section Results: Statistical significance not reported. Suggested fix: add p-values or confidence intervals."
  ],
  "minor_issues": [
    "Introduction: Missing roadmap paragraph at end of section.",
    "Related Work: Paragraph 3 is descriptive rather than argumentative."
  ],
  "required_revisions": [
    "Add at least two baseline comparisons in the experiments section.",
    "Include a reproducibility checklist as an appendix.",
    "Fix all undefined LaTeX references (\\ref without matching \\label)."
  ],
  "lessons_learned": "Papers score higher when they include real-world datasets alongside synthetic benchmarks. Ablation studies tied to the hypothesis are essential for methodology scores above 7.",
  "final_verdict": "major_revision"
}
```

## Scoring Rubric

| Score | Meaning |
|-------|---------|
| 9–10 | Publication-ready |
| 7–8 | Minor revisions needed |
| 5–6 | Major revisions needed |
| 3–4 | Fundamental issues |
| 0–2 | Reject |

## Issue Format

Each issue MUST follow this format:
> "Section X: [specific problem]. Suggested fix: [specific action]."

**NOT**: "The paper needs improvement." (too vague — not actionable)

## Quality Criteria

- All 6 score dimensions present and in range 0–10
- Major issues are actionable with specific section references
- Required revisions are implementable (not vague suggestions)
- Lessons learned provides transferable insight for future cycles
- Final verdict consistent with overall_score

## Eval Assertions

- **json_parseable**: output is valid JSON
- **has_all_score_keys**: all 6 score keys present (novelty, methodology, clarity, reproducibility, figures, coherence)
- **scores_in_range**: all scores between 0 and 10
- **major_issues_actionable**: each issue contains ":" (section:problem format)
- **required_revisions_not_empty**: len(required_revisions) >= 1
- **verdict_valid**: final_verdict in ["accept", "minor_revision", "major_revision"]
