---
name: aras-coherence-reviser
version: 2
description: >
  Revises a LaTeX paper for coherence and applies reviewer feedback. Triggers when ARAS needs to revise paper.tex after a review round.
performance_history:
    - cycle: 1
      paper_score: 6.0
      component_score: 7.5
      topic: "Can background music tempo influence decision fatigue? A stu"
      version: 2
    - cycle: 1
      paper_score: 3.0
      component_score: 3.0
      topic: "Uncertainty Quantification in Deep Neural Networks via Ensem"
      version: 2
  - cycle: 1
    topic: "Uncertainty Quantification in Deep Neural Networks via Ensemble Methods and Bayesian Approximations"
    score: 3.0
    issues: ["Empty Abstract section with no content", "Empty Introduction missing motivation and contributions", "Methodology section contained only placeholder header 'System Architecture' with no technical content, equations, or algorithms"]
---

# Coherence Reviser Skill

## Core Task
Given paper.tex + reviewer feedback JSON, produce a **STRICT JSON** response with:
- **revised_tex** — complete revised LaTeX source
- **coherence_score** — float 0–10
- **issues_found** — int
- **issues_fixed** — int
- **changes_summary** — string (bullet list of what changed)

## Content Completeness Requirements (CRITICAL)
Before returning, verify these minimum content standards to prevent empty sections:

### Abstract Requirements
- Must contain 150-200 words of substantive text
- Must include: (1) problem statement, (2) proposed approach/methods, (3) key empirical results with specific metrics, (4) significance/contribution
- Must NOT be empty or contain only placeholder text like "[Abstract here]" or "TBD"

### Introduction Requirements
- Must contain 2-3 paragraphs of substantive prose (>150 words total)
- Paragraph 1: Motivation and importance of the problem
- Paragraph 2: Specific gaps in current approaches (ensembles, Bayesian methods, etc.)
- Paragraph 3: Explicit numbered or bulleted list of 3-4 technical contributions
- Must NOT contain only a heading or placeholder text

### Methodology Requirements
- Must contain substantive technical content (>200 words)
- Must include: mathematical formulations (at least one equation environment), algorithmic details (pseudocode or step-by-step procedure), and implementation specifics (network architectures, hyperparameters)
- Must NOT contain only section headers (e.g., "System Architecture") without following explanatory text and technical details
- Every subsection header must be followed by at least one paragraph of explanation before the next header

## Revision Checklist (apply ALL)
- [ ] Abstract mentions all 3+ contributions from introduction and contains 150-200 words
- [ ] Every experiment mentioned in methodology appears in results
- [ ] Conclusion restates contributions from introduction (not new claims)
- [ ] All `\cite{}` keys have matching entries in references/bibliography
- [ ] All `\ref{}` labels have matching `\label{}` definitions
- [ ] Required revisions from reviewer feedback applied
- [ ] Section transitions present (last sentence of each section introduces the next)
- [ ] No passive voice in methodology (prefer "we propose" over "it is proposed")
- [ ] No orphan figures (every figure referenced in text with `Figure~\ref{}`)
- [ ] No duplicate content between sections
- [ ] **No empty sections**: Abstract, Introduction, Methodology, Results, Conclusion each contain >100 words of substantive text
- [ ] **No placeholder headers**: Every `\section{}` and `\subsection{}` is immediately followed by explanatory prose (not another header or empty line)
- [ ] **Technical content present**: Methodology contains at least one equation environment (`\begin{equation}`, `\[`, or `$$`) and describes algorithms/implementation details

## Output Requirements
Return the **COMPLETE revised LaTeX** document. Not a diff. Not a summary. The full `paper.tex` with ALL changes applied. The `revised_tex` string must start with `\documentclass` and end with `\end{document}`.

**Validation Step**: Before finalizing, scan `revised_tex` to ensure:
1. Abstract environment contains text between `\begin{abstract}` and `\end{abstract}`
2. Introduction section contains text after `\section{Introduction}` and before `\section{` (next section)
3. Methodology section contains equations or algorithm environments
4. No section consists solely of LaTeX structural commands (headers, labels, comments)

## Quality Criteria
- revised_tex is a complete compilable LaTeX document
- issues_fixed <= issues_found (can't fix more than found)
- changes_summary explains each change made
- Reviewer's required_revisions are addressed (or explained if impossible)
- Abstract word count between 150-200 words
- Methodology contains mathematical equations or algorithmic pseudocode

## Eval Assertions
- **returns_complete_latex**: output contains `\documentclass`
- **coherence_score_present**: coherence_score is a float
- **issues_fixed_lte_found**: issues_fixed <= issues_found
- **no_broken_refs**: count of `\ref{` approximately matches `\label{`
- **has_changes_summary**: changes_summary is non-empty string
- **no_empty_sections**: Abstract, Introduction, and Methodology sections each contain >50 words of content (detected by word count between section boundaries)
- **methodology_has_equations**: Methodology section contains at least one equation environment (`\begin{equation}`, `\[`, `$$`, or `\begin{align}`)
- **introduction_lists_contributions**: Introduction section contains a numbered or bulleted list (detected by `\begin{itemize}`, `\begin{enumerate}`, or pattern `\n\d+\.\s` or `\n-\s` within the Introduction section)