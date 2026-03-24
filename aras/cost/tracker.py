from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from aras.cost.pricing import PricingTier, get_pricing


@dataclass(frozen=True)
class CostLine:
    """A single recorded LLM call usage for ledgers and reporting."""

    ts: str
    cycle: int | None
    agent_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class CostReport:
    """Rollup cost and token usage for a single cycle."""

    cycle: int
    total_cost_usd: float
    by_agent: dict[str, float]
    by_provider: dict[str, float]
    token_breakdown: dict[str, int]
    tokens_input: int
    tokens_output: int
    tokens_total: int


class CostTracker:
    """Thread-safe cost tracker with JSONL persistence."""

    def __init__(self, *, logs_dir: Path) -> None:
        self._lock = threading.Lock()
        self._logs_dir = logs_dir
        self._ledger_path = self._logs_dir / "cost_log.jsonl"

        # Running totals for the entire run.
        self._total_cost_usd: float = 0.0
        self._total_tokens_input: int = 0
        self._total_tokens_output: int = 0
        self._total_tokens_total: int = 0

        # Cycle-local accumulators (reset at begin_cycle()).
        self._cycle: int | None = None
        self._cycle_cost_usd: float = 0.0
        self._cycle_tokens_input: int = 0
        self._cycle_tokens_output: int = 0
        self._cycle_tokens_total: int = 0
        self._cycle_by_agent: dict[str, float] = {}
        self._cycle_by_provider: dict[str, float] = {}

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return float(self._total_cost_usd)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return int(self._total_tokens_total)

    def begin_cycle(self, cycle: int) -> None:
        """Reset cycle accumulators."""
        with self._lock:
            self._cycle = int(cycle)
            self._cycle_cost_usd = 0.0
            self._cycle_tokens_input = 0
            self._cycle_tokens_output = 0
            self._cycle_tokens_total = 0
            self._cycle_by_agent = {}
            self._cycle_by_provider = {}

    def record(
        self,
        *,
        agent_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        ts: str | None = None,
    ) -> None:
        """Record a single call and append to JSONL ledger."""
        with self._lock:
            input_tokens = int(input_tokens)
            output_tokens = int(output_tokens)
            total_tokens = input_tokens + output_tokens

            tier: PricingTier = get_pricing(provider, model)
            cost_usd = (input_tokens / 1_000_000.0) * tier.input_per_million_usd + (output_tokens / 1_000_000.0) * tier.output_per_million_usd

            now = ts or datetime.now(timezone.utc).isoformat()
            line = CostLine(
                ts=now,
                cycle=self._cycle,
                agent_id=str(agent_id),
                provider=str(provider),
                model=str(model),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=float(cost_usd),
            )

            # Persist ledger line.
            self._logs_dir.mkdir(parents=True, exist_ok=True)
            with self._ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line.__dict__, ensure_ascii=False) + "\n")

            # Update totals.
            self._total_cost_usd += float(cost_usd)
            self._total_tokens_input += input_tokens
            self._total_tokens_output += output_tokens
            self._total_tokens_total += total_tokens

            # Update cycle totals.
            self._cycle_cost_usd += float(cost_usd)
            self._cycle_tokens_input += input_tokens
            self._cycle_tokens_output += output_tokens
            self._cycle_tokens_total += total_tokens

            self._cycle_by_agent[line.agent_id] = float(self._cycle_by_agent.get(line.agent_id, 0.0) + float(cost_usd))
            self._cycle_by_provider[line.provider] = float(self._cycle_by_provider.get(line.provider, 0.0) + float(cost_usd))

    def end_cycle(self, cycle: int, *, budget_ceiling_usd: float | None = None) -> tuple[CostReport, bool]:
        """Finalize cycle report and return (report, budget_exceeded)."""
        with self._lock:
            if self._cycle is None or int(cycle) != int(self._cycle):
                # Be forgiving: allow end_cycle without strict begin_cycle order.
                self._cycle = int(cycle)

            report = CostReport(
                cycle=int(cycle),
                total_cost_usd=float(self._cycle_cost_usd),
                by_agent=dict(self._cycle_by_agent),
                by_provider=dict(self._cycle_by_provider),
                token_breakdown={"input": int(self._cycle_tokens_input), "output": int(self._cycle_tokens_output), "total": int(self._cycle_tokens_total)},
                tokens_input=int(self._cycle_tokens_input),
                tokens_output=int(self._cycle_tokens_output),
                tokens_total=int(self._cycle_tokens_total),
            )

            budget_exceeded = False
            if budget_ceiling_usd is not None:
                budget_exceeded = report.total_cost_usd >= float(budget_ceiling_usd)

            # Prepare for next cycle by clearing cycle accumulators.
            self._cycle = None
            return report, budget_exceeded

    def budget_remaining_usd(self, *, budget_ceiling_usd: float | None) -> float | None:
        """Compute remaining budget based on total_cost_usd."""
        if budget_ceiling_usd is None:
            return None
        remaining = float(budget_ceiling_usd) - self.total_cost_usd
        return remaining


DEFAULT_LEDGER_BASENAME: Final[str] = "cost_log.jsonl"

