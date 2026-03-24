# Functional Truth Contract

This document defines the non-negotiable functional checks that prove ARAS behavior is real and connected end-to-end.

## Novelty (Strict Evidence Gate)

- Pivot only occurs when `novelty.gate_passed == true`.
- Low novelty score alone is not enough to pivot.
- Gate requires all:
  - `novelty_score < NOVELTY_PIVOT_MAX_SCORE`
  - `confidence >= NOVELTY_MIN_CONFIDENCE`
  - `validated_evidence_count >= NOVELTY_MIN_VALIDATED_EVIDENCE`
  - `len(evidence_sources) >= NOVELTY_MIN_EVIDENCE_SOURCES`
  - non-empty `selected_angle`
- If blocked, `gate_reason` must be set and emitted in logs/events.

Evidence:
- `logs/novelty_evidence_cycle{n}.json`
- orchestrator novelty log events

## Figures (Quality Gate)

- Low-confidence diagnostic figures are generated for debugging/UI only.
- Low-confidence figures never appear in paper LaTeX (`analysis.figures_latex`).
- If all runs degraded, orchestrator reruns experiments up to configured cap.

Evidence:
- `paper/figures/*`
- `analysis.figure_quality_summary`
- orchestrator rerun events

## Improvement Tracking

- Every cycle appends one line to `logs/cycle_quality.jsonl`.
- Row includes novelty quality, experiment health, figure quality, review/paper score, and `improvement_index`.

Evidence:
- `logs/cycle_quality.jsonl`

## UI/API Connectivity

- `/api/figures` reflects actual files under `paper/figures` and caption metadata.
- `/api/quality` returns latest row from `logs/cycle_quality.jsonl`.
- WebSocket `state` and `log` events reflect real orchestrator state transitions.

Evidence:
- API responses in tests
- WebSocket event tests

## Anti-Theater Rule

- If artifact/log does not exist, UI should not claim success for that subsystem.
- “Ready” states require backing file/event/state proof.
