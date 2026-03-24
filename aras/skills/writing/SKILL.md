---
name: aras-paper-writer
version: 2
description: >
  Drafts individual IEEE paper sections from research artifacts. Triggers when ARAS needs to write any paper section. Enforces validation checkpoints to ensure all sections contain substantive technical content; empty templates or placeholder headers are prohibited.
performance_history:
    - cycle: 1
      paper_score: 6.0
      component_score: 7.5
      topic: "Can background music tempo influence decision fatigue? A stu"
      version: 0
    - cycle: 1
      paper_score: 3.0
      component_score: 4.0
      topic: "Uncertainty Quantification in Deep Neural Networks via Ensem"
      version: 0
  - cycle: 1
    topic: "Uncertainty Quantification in Deep Neural Networks via Ensemble Methods and Bayesian Approximations"
    score: 4.0/10
    issues: ["Empty abstract", "Missing introduction content", "Methodology contained only headers", "No empirical results", "Related work was descriptive list not argumentative"]
---

# Paper Writer Skill

## Core Task
Given section name + research plan + scraped items + experiment results, write **ONE** paper section in LaTeX. **Validation Checkpoint**: Before outputting, verify the section contains substantive technical content (equations, arguments, or empirical numbers), not just headers or placeholder text. Empty templates cannot be evaluated for scientific merit.

## Section-Specific Guidance

### abstract
**Mandatory**: 150–250 words. Structure: Problem → gap → method → result → impact. 
- Must include specific quantitative results if available (e.g., "reduces calibration error by X%")
- No citations allowed
- **Anti-failure rule**: If word count < 100, expand with specific technical details about the method

### introduction
**Mandatory structure**:
1. **Hook** (1 paragraph): Concrete motivation with real-world stakes
2. **Problem Statement** (1 paragraph): Formal definition of the technical gap
3. **Literature Gap** (1 paragraph): Specific limitations of prior work (cite 3–4 papers)
4. **Contributions** (bullet list): Exactly 3–4 technical contributions, each starting with "We" and including quantitative claims where possible
5. **Roadmap**: "The remainder of this paper is organized as follows..."

**Anti-failure rule**: Never output only section headers; each paragraph must contain argumentative content.

### related_work
**ARGUMENTATIVE, not descriptive.** Every paragraph must follow the template:
> "X et al. \cite{} proposed Y, achieving Z. However, this approach fails at W because V. Our work addresses W by..."

- Minimum 8 citations across 3–4 thematic paragraphs
- Group by technical approach (e.g., "Ensemble Methods", "Bayesian Approximations"), not chronologically
- Every paragraph must contain a contrastive keyword: however, although, whereas, limited, fails, unable
- **Anti-failure rule**: Raw itemized lists of papers are prohibited; synthesize into arguments

### methodology
**Validation Checkpoint**: Section must contain:
- Formal mathematical definitions (equations) OR algorithm pseudocode (algorithmic environment)
- Specific implementation details (architectures, hyperparameters)
- Justification for every design choice

**Structure**:
1. Preliminaries/Notation (define all symbols before use)
2. Proposed Method (equations + algorithm if applicable)
3. Implementation Details

**Anti-failure rule**: Subsection headers must be followed by at least one paragraph of technical content or equations; never output standalone headers.

### experiments
**Prioritize real benchmarks**. Setup must include:
- Dataset specifications with sample sizes and splits
- Baseline descriptions with citations
- Metrics with mathematical definitions
- Computational environment (hardware/software)

**Anti-failure rule**: Include ablations tied to the hypothesis from the research plan.

### results
**Interpret numbers — never just describe them**:
- ✅ "Model A outperforms B by 3.2% on F1 (p < 0.05), reducing calibration error from 0.15 to 0.08"
- ❌ "Table 1 shows the results"

**Requirements**:
- Must reference specific numerical values from experiment_results
- Include statistical significance where applicable
- Tables use `\begin{table}` with `\toprule`, `\midrule`, `\bottomrule`
- **Anti-failure rule**: Empty tables or placeholder text ("results will be added") are prohibited; populate with actual metrics from provided experiment results

### discussion
**Structure**:
1. Limitations (honest technical constraints)
2. Future Work (3 specific directions)
3. Broader Impact (societal implications)

### conclusion
- Restate exactly 3 contributions from introduction (rephrased)
- No new information, no citations
- Final sentence on significance

## Output Format
Return **raw LaTeX** for the section. Include `\section{...}` header. No `\documentclass` or preamble.

**Content Validation**: Before finalizing, verify:
1. No Jinja2 placeholders (`{{` or `}}`) remain
2. All section headers have substantive content following them
3. Results section contains actual numeric values from experiment results
4. Related work paragraphs contain contrastive arguments, not just descriptions

## Quality Criteria
- No unfilled Jinja2 placeholders (`{{` or `}}`)
- Related work: argumentative not descriptive; every paragraph critiques prior work
- Every section ends with a transition sentence to the next section
- LaTeX compiles (balanced braces, valid commands)
- Tables use booktabs formatting (`\toprule`, `\midrule`, `\bottomrule`)
- **Substantive content**: No section may contain only headers, outlines, or placeholder text
- **Empirical grounding**: Results must include quantitative values from provided experiment results, not hypothetical values

## Eval Assertions
- **no_placeholders**: `{{` not in output and `}}` not in output
- **has_citations**: `\cite{` in output (for all sections except abstract/conclusion)
- **latex_balanced**: count of `\begin` equals count of `\end`
- **word_count_reasonable**: 200 < word count < 1500
- **no_empty_sections**: No standalone `\subsection{...}` or `\subsubsection{...}` without following content (minimum 50 words per subsection)
- **contains_quantitative_results**: Results section must contain numeric values (regex: `\d+\.\d+` or `\d+%`) from experiment_results
- **argumentative_related_work**: Related work must contain at least 3 contrastive keywords (however, although, whereas, limited, fails, unable) and no bullet-point-only citations