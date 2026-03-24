from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


AgentStatus = Literal["IDLE", "WORKING", "DONE", "ERROR"]


@dataclass
class AgentState:
    """Per-agent UI state snapshot."""

    status: AgentStatus = "IDLE"
    last_update: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "last_update": self.last_update, "detail": self.detail}


@dataclass
class PipelineStep:
    """Progress tracking for the UI."""

    name: str
    progress: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "progress": float(self.progress)}


class FinalReport(BaseModel):
    """Final report for a completed ARAS cycle run."""

    topic: str
    cycles: int
    paper_score: Optional[float]
    github_url: Optional[str]
    tokens_used: int
    elapsed_seconds: float
    outputs: Dict[str, str]

    class Config:
        arbitrary_types_allowed = True

    def to_json(self, *, indent: int | None = None) -> str:
        """Compatibility helper for legacy callers."""
        # Exclude None so UI/clients don't have to handle nulls everywhere.
        return self.model_dump_json(indent=indent, exclude_none=True)


class NoveltyResult(BaseModel):
    """Result of novelty check and potential pivot."""

    original_topic: str
    selected_angle: str
    novelty_score: float
    competing_papers: List[Dict[str, Any]] = Field(default_factory=list)
    pivot_reason: Optional[str] = None
    confidence: float = 0.0
    evidence_count: int = 0
    validated_evidence_count: int = 0
    evidence_sources: List[str] = Field(default_factory=list)
    gate_passed: bool = False
    gate_reason: Optional[str] = None


class Citation(BaseModel):
    """Validated citation object."""

    source: str
    title: str
    abstract: str
    authors: List[str]
    published: Optional[str] = None
    url: str
    doi: Optional[str] = None
    citation_count: int = 0
    has_code: bool = False
    code_url: Optional[str] = None
    relevance: float = 0.0
    validated: bool = False
    retracted: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricEvent(BaseModel):
    """Live metric emitted from experiments."""

    experiment: str
    key: str
    value: float
    step: int
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PaperDiff(BaseModel):
    """Structured diff between paper revisions."""

    round: int
    sections_changed: List[str]
    lines_added: int
    lines_removed: int
    summary: str


class FailureRecord(BaseModel):
    """Structured failure taxonomy record."""

    type: str
    agent_id: str
    cycle: int
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved: bool = False
    resolution: Optional[str] = None


class CostLine(BaseModel):
    """Single line of cost usage."""

    agent_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CostReport(BaseModel):
    """Per-cycle cost report."""

    cycle: int
    total_cost_usd: float
    by_agent: Dict[str, float]
    by_provider: Dict[str, float]
    token_breakdown: Dict[str, int]
    budget_ceiling: float
    budget_remaining: float


class HFPublishResult(BaseModel):
    """Results of Hugging Face publishing."""

    dataset_url: Optional[str] = None
    paper_url: Optional[str] = None
    space_url: Optional[str] = None


class ApprovalPayload(BaseModel):
    """Payload sent to the external approval webhook."""

    topic: str
    cycle: int
    paper_score: float
    novelty_score: float
    abstract: str
    sections_completed: List[str]
    github_draft_url: Optional[str] = None
    cost_usd: float
    timestamp: str


class ApprovalDecision(BaseModel):
    """Decision returned to /api/approval."""

    approved: bool
    note: Optional[str] = None
