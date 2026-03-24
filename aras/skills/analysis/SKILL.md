---
name: aras-results-analyst
version: 1
description: >
  Analyzes experiment results and produces narrative + tables.
  Triggers when ARAS needs to interpret results.json after experiments run.
performance_history:
    - cycle: 1
      paper_score: 6.0
      component_score: 5.5
      topic: "Can background music tempo influence decision fatigue? A stu"
      version: 0
    - cycle: 1
      paper_score: 3.0
      component_score: 1.0
      topic: "Uncertainty Quantification in Deep Neural Networks via Ensem"
      version: 0
---

# Results Analyst Skill

## Core Task

Given `results.json` + research plan, produce a **STRICT JSON** analysis:

- **table_markdown** — markdown table of all metrics
- **narrative** — 2–3 paragraph interpretation of results
- **lessons_learned** — string (what worked, what didn't)
- **best_experiment** — string (name of best-performing experiment)
- **key_finding** — string (the single most important result, 1 sentence)

## Narrative Guidelines

- Compare results against baselines **explicitly** ("Model A outperforms baseline B by X%")
- State statistical significance if sample size > 30
- Never just describe numbers — **interpret** them
- Connect findings back to the research hypothesis
- Mention any unexpected results and hypothesize why

## Output Format

```json
{
  "table_markdown": "| Experiment | Accuracy | F1 | Runtime |\n|---|---|---|---|\n| baseline | 0.72 | 0.68 | 12.3s |\n| proposed | 0.81 | 0.79 | 14.1s |",
  "narrative": "Our proposed method achieves an accuracy of 0.81, outperforming the baseline by 12.5%...",
  "lessons_learned": "Temperature scaling proved most effective when combined with...",
  "best_experiment": "proposed_with_calibration",
  "key_finding": "Platt scaling reduces expected calibration error by 38% compared to raw softmax probabilities."
}
```

## Quality Criteria

- Table must include ALL metrics from results.json
- Narrative must reference at least 2 specific numeric values
- Key finding must be ONE sentence
- Lessons learned must be actionable for future cycles

## Eval Assertions

- **has_all_keys**: all 5 required keys present
- **table_is_markdown**: "|" in table_markdown
- **narrative_has_comparison**: contains "outperform" or "better" or "worse" or "compared" or "improvement"
- **key_finding_one_sentence**: len(key_finding.split('.')) <= 2
- **best_experiment_not_empty**: best_experiment is non-empty string
