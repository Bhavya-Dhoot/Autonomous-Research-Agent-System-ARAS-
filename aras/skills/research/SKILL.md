---
name: aras-research-planner
version: 3
description: >
  Generates a structured research plan from a topic string with mandatory section content briefs to prevent empty template sections. Enforces substantive technical details in methodology, validation checkpoints for autonomous generation, and ablation studies tied to the hypothesis. Explicitly validates experimental design terminology (distinguishing between-subjects, within-subjects, and factorial designs) and requires control conditions for sensory manipulation studies.
performance_history:
    - cycle: 1
      paper_score: 6.0
      component_score: 6.0
      topic: "Can background music tempo influence decision fatigue? A stu"
      version: 3
  - cycle: 1
    paper_score: 3.0
    component_score: 3.0
    topic: "Uncertainty Quantification in Deep Neural Networks via Ensem"
    version: 2
    issues:
      - "Section Abstract: Completely empty with no summary of contributions, methods, or findings"
      - "Section Introduction: Missing motivation, problem statement, and contribution overview"
      - "Section Methodology: Contains only subsection headers with no technical content, equations, or algorithmic details"
  - cycle: 1
    topic: "Can background music tempo influence decision fatigue? A study on cognitive depletion rates in prolonged online shopping sessions"
    component_score: 6.0
    version: 2
    issues:
      - "Section Methodology: Incorrect use of 'factorial design' terminology (only one independent variable manipulated). Suggested fix: correct to 'between-subjects design' or add a second factor to make it factorial."
---

# Research Planner Skill

## Core Task
Given a research topic, produce a **STRICT JSON** plan with:
- **hypothesis** — 1 sentence, falsifiable claim with specific technical mechanism (e.g., "via gradient covariance estimation" or "through 60-90 BPM tempo modulation")
- **questions** — 3–5 specific, falsifiable research questions answerable by the experiments
- **experiments** — 3 experiment names including at least one ablation study tied to the hypothesis mechanism (for behavioral studies: include manipulation checks and control conditions)
- **metrics** — list of measurable outcomes with units (e.g., "wall_time_seconds", "expected_calibration_error", "decision_latency_ms", "cognitive_depletion_score")
- **outline** — list of paper section titles (standard academic structure)
- **section_briefs** — **REQUIRED** object mapping each outline title to a 2-3 sentence description of specific technical content to be included. Must contain:
  - For computational studies: equations, algorithms, key results, architectures
  - For experimental/behavioral studies: design type, variables, control conditions, participant demographics, statistical tests
- **keywords** — 8–12 search terms including both general and specific technical terms
- **domain** — one of: `nlp`, `computer_vision`, `reinforcement_learning`, `general_ml`, `theory`, `systems`, `bioinformatics`, `experimental_psychology`, `human_computer_interaction`, `other`

## Output Format
Return **JSON only**. No markdown. No preamble. No explanation.

```json
{
  "hypothesis": "We hypothesize that ...",
  "questions": ["Q1", "Q2", "Q3"],
  "experiments": ["exp1_baseline", "exp2_ablation_mechanism", "exp3_robustness"],
  "metrics": ["accuracy", "f1", "runtime_seconds"],
  "outline": ["Abstract", "Introduction", "Related Work", "Methodology", "Experiments", "Results", "Discussion", "Conclusion"],
  "section_briefs": {
    "Abstract": "150-200 words summarizing the problem, approach, key empirical results, and significance.",
    "Introduction": "3 paragraphs: (1) Motivation and problem statement, (2) Specific gap in literature, (3) 3-4 specific contributions.",
    "Methodology": "Technical details: mathematical formulations or experimental design specifications.",
    "Experiments": "Implementation details with dataset/participant specifications and statistical methodology.",
    "Results": "Tables and figures showing primary outcomes and ablation analyses.",
    "Discussion": "Interpretation of results, limitations, and future work.",
    "Conclusion": "Summary of findings and practical implications."
  },
  "keywords": ["keyword1", "keyword2", "...", "keyword10"],
  "domain": "general_ml"
}
```

## Quality Criteria
- Hypothesis **must be falsifiable** — contains "we hypothesize", "we propose", or "we demonstrate" AND specifies the technical mechanism (e.g., "via diagonal Hessian approximation", "using deep ensembles of size M≥5", "through 60-90 BPM tempo modulation")
- **Section briefs are mandatory** and must contain specific technical markers:
  - Computational: mathematical notation (e.g., "L(θ)", "∇"), algorithm names (e.g., "SGD with momentum"), concrete architectural details (e.g., "ResNet-18 backbone")
  - Experimental: design type specification (between-subjects, within-subjects, or factorial), independent/dependent variable definitions, control conditions (e.g., "silence/no-stimulus baseline"), sample size, statistical tests (e.g., "mixed-effects ANOVA")
- **Experimental design accuracy**: 
  - Single-factor designs must use "between-subjects" or "within-subjects" terminology, never "factorial"
  - Factorial designs require explicit specification of ≥2 independent variables with their respective levels (e.g., "2×3 factorial: tempo (slow/fast) × task duration (short/medium/long)")
  - Sensory manipulation studies must include control condition description (e.g., "silence/no-music baseline") to isolate effects from general environmental factors
- Abstract brief must specify word count (150-200) and mention contributions, methods, and key findings
- Introduction brief must explicitly mention "motivation", "gap", and list 3-4 specific contributions
- Methodology brief must include at least one equation description (computational) or complete experimental design specification (behavioral)
- Experiments must include at least one ablation study (computational) or manipulation check (behavioral) targeting the hypothesis mechanism
- Keywords must include both **general terms** AND **specific technical terms** (e.g., both "machine learning" and "gradient boosting calibration", or both "cognitive psychology" and "decision fatigue")
- Domain must match the most prominent keywords
- Output must be valid parseable JSON — no trailing commas, no comments, no markdown code blocks

## Example (Computational)
**Input**: "Calibration of Confidence Scores in Text Classification Models"
**Output**: 
```json
{
  "hypothesis": "We hypothesize that temperature scaling with Platt calibration outperforms raw softmax probabilities for confidence calibration in transformer-based text classifiers via learned temperature parameters T optimized on validation NLL.",
  "questions": [
    "How does temperature scaling compare to isotonic regression for expected calibration error reduction?",
    "What is the impact of dataset size (N<1000 vs N>10000) on calibration quality measured by Brier score?",
    "Do multilingual models exhibit worse calibration than monolingual models as measured by reliability diagram deviation?"
  ],
  "experiments": [
    "baseline_calibration_comparison",
    "ablation_temperature_parameter_t",
    "cross_domain_calibration_transfer"
  ],
  "metrics": ["expected_calibration_error", "brier_score", "reliability_diagram_auc", "accuracy", "calibration_time_ms"],
  "outline": ["Abstract", "Introduction", "Related Work", "Methodology", "Experiments", "Results", "Discussion", "Conclusion"],
  "section_briefs": {
    "Abstract": "150 words: Problem of miscalibrated transformers in NLP, proposed temperature scaling solution, key results showing 40% ECE reduction on GLUE, significance for trustworthy deployment.",
    "Introduction": "Motivation: deployed NLP models require calibrated confidence for medical/legal applications. Gap: existing calibration ignores transformer-specific attention patterns. Contributions: (1) Temperature scaling for BERT variants, (2) Dataset size sensitivity analysis, (3) Cross-lingual calibration transfer, (4) Computational overhead analysis.",
    "Methodology": "Temperature scaling: p_calibrated = softmax(z/T) where T is learned via grid search on validation set. Platt scaling: logistic regression on logits. Implementation: PyTorch 2.0, HuggingFace Transformers, batch size 32.",
    "Experiments": "Dataset: SST-2, IMDB, MultiNLI. Baselines: uncalibrated, histogram binning, isotonic regression. Ablation: T fixed vs learned, effect of validation set size on T estimation.",
    "Results": "Table 1: ECE and Brier score comparisons. Figure 1: Reliability diagrams before/after calibration. Ablation: ECE vs temperature parameter T showing optimal T≈1.5.",
    "Discussion": "Analysis of why temperature scaling outperforms binning on small datasets. Limitation: assumes model is already reasonably accurate. Future work: learned per-class temperatures.",
    "Conclusion": "Temperature scaling provides efficient calibration for transformers with minimal overhead."
  },
  "keywords": ["confidence calibration", "text classification", "temperature scaling", "Platt scaling", "expected calibration error", "transformer", "softmax probability", "isotonic regression", "reliability diagram", "Brier score"],
  "domain": "nlp"
}
```

## Example (Experimental)
**Input**: "Can background music tempo influence decision fatigue? A study on cognitive depletion rates in prolonged online shopping sessions"
**Output**:
```json
{
  "hypothesis": "We hypothesize that slow-tempo background music (60-90 BPM) reduces decision fatigue compared to fast-tempo music (120-140 BPM) via decreased cognitive arousal and lower information processing demands during prolonged choice tasks.",
  "questions": [
    "Does slow-tempo music significantly reduce decision reaction time compared to fast-tempo music and silence control?",
    "What is the effect of music tempo on subjective cognitive depletion scores measured by NASA-TLX?",
    "At what point in the shopping session (minutes elapsed) does tempo effects on decision quality become significant?"
  ],
  "experiments": [
    "between_subjects_tempo_manipulation",
    "manipulation_check_arousal_validation",
    "dose_response_tempo_duration"
  ],
  "metrics": ["decision_latency_ms", "nasa_tlx_score", "choice_consistency_rate", "arousal_valence_rating", "session_completion_time"],
  "outline": ["Abstract", "Introduction", "Related Work", "Methodology", "Experiments", "Results", "Discussion", "Conclusion"],
  "section_briefs": {
    "Abstract": "180 words: Problem of decision fatigue in e-commerce, proposed music tempo intervention, key results showing 15% reduction in cognitive depletion with slow music (n=90), significance for interface design.",
    "Introduction": "Motivation: online shopping fatigue leads to suboptimal decisions. Gap: unclear if ambient music tempo affects cognitive depletion. Contributions: (1) Causal link between BPM and decision fatigue, (2) Optimal tempo range identification, (3) Temporal dynamics of fatigue onset, (4) Practical interface recommendations.",
    "Methodology": "Between-subjects design with three conditions: slow tempo (60-90 BPM), fast tempo (120-140 BPM), and silence control. Independent variable: music tempo. Dependent variables: decision latency, choice consistency, NASA-TLX scores. Participants: N=90 (30 per condition), aged 18-35. Statistical analysis: one-way ANOVA with Bonferroni correction, mixed-effects models for temporal effects.",
    "Experiments": "Task: 45-minute simulated shopping with 120 product choices. Manipulation check: SAM scale arousal ratings post-session. Control: ambient noise <40dB for silence condition. Counterbalancing: product presentation order randomized.",
    "Results": "Table 1: Mean decision latency and NASA-TLX by condition. Figure 1: Fatigue accumulation curves over time. Manipulation check: arousal ratings confirm tempo manipulation (p<0.001).",
    "Discussion": "Mechanism: slow music reduces cognitive load via parasympathetic activation. Limitation: single-session design, ecological validity. Future work: individual musical preference moderators.",
    "Conclusion": "Slow-tempo music mitigates decision fatigue in prolonged shopping; recommend 60-80 BPM for retail environments."
  },
  "keywords": ["decision fatigue", "cognitive depletion", "background music", "tempo", "between-subjects design", "online shopping", "NASA-TLX", "cognitive load", "consumer psychology", "arousal"],
  "domain": "experimental_psychology"
}
```

## Eval Assertions
- **json_parseable**: output is valid JSON
- **has_all_keys**: all 8 required keys present (hypothesis, questions, experiments, metrics, outline, section_briefs, keywords, domain)
- **hypothesis_falsifiable**: contains "hyp