---
name: aras-experiment-coder
version: 1
description: >
  Designs and writes domain-appropriate experiment Python code.
  Triggers when ARAS needs to generate experiment scripts.
performance_history:
    - cycle: 1
      paper_score: 6.0
      component_score: 5.0
      topic: "Can background music tempo influence decision fatigue? A stu"
      version: 0
    - cycle: 1
      paper_score: 3.0
      component_score: 0.0
      topic: "Uncertainty Quantification in Deep Neural Networks via Ensem"
      version: 0
---

# Experiment Coder Skill

## Core Task

Given research plan + domain classification, write runnable Python experiments that:

- Use **real publicly available datasets** (sklearn fallbacks required)
- Emit `METRIC_JSON` lines to stdout for live streaming
- Write `results.json` and `loss.png`/`loss.pdf` on completion
- Complete in **< 300 seconds** on CPU
- Handle timeout gracefully (partial results + TIMEOUT_SAFE flag)

## Domain → Experiment Mapping

| Domain | Experiments |
|--------|------------|
| nlp | text_classification, sentiment_analysis, tokenizer_benchmark |
| computer_vision | image_classification, feature_extraction, augmentation |
| reinforcement_learning | bandit_comparison, policy_gradient, q_learning |
| general_ml | cross_validation, hyperparameter_sensitivity, convergence |
| theory | algorithm_complexity, numerical_bounds, information_theory |

## Code Quality Requirements

Every experiment file MUST include:

```python
import json
import signal
from pathlib import Path
import numpy as np

np.random.seed(42)  # reproducibility
```

### METRIC_JSON Format

Print metrics to stdout for live streaming:
```python
print(f'METRIC_JSON {json.dumps({"experiment": "exp_name", "metric": "loss", "value": 0.5, "step": 1})}', flush=True)
```

### results.json Format

```json
{
  "experiment": "baseline_benchmark",
  "domain": "general_ml",
  "metrics": {"accuracy": 0.82, "f1": 0.79, "runtime_seconds": 45.2},
  "runtime_seconds": 45.2
}
```

### Timeout Guard

Always include a timeout guard:
```python
import signal

def _timeout_handler(signum, frame):
    # Save partial results
    Path("results.json").write_text(json.dumps({"status": "TIMEOUT_SAFE", "partial": True}))
    raise SystemExit(0)

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(280)  # 280s safety margin
```

Or use `threading.Timer` on Windows where `signal.SIGALRM` is unavailable.

### sklearn Fallback

Always include a fallback for missing datasets:
```python
try:
    from some_library import load_data
    X, y = load_data()
except Exception:
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
```

## Quality Criteria

- All randomness seeded with `np.random.seed(42)` or equivalent
- Code runs without network access (datasets must be local or generatable)
- No hardcoded absolute paths
- Results written to current working directory

## Eval Assertions

- **valid_python**: `compile(code, '<string>', 'exec')` succeeds
- **has_metric_json**: "METRIC_JSON" in code
- **has_results_write**: "results.json" in code
- **has_timeout_guard**: "signal" in code or "threading.Timer" in code
- **has_sklearn_fallback**: "except" in code and "sklearn" in code
- **has_seed**: "seed(42)" in code
