from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator
import threading

from aras.config import Settings
from aras.types import AgentState, ApprovalPayload, FinalReport, PipelineStep
from aras.utils.fs import safe_write_text
from aras.utils.logging import get_logger
from aras.cost.tracker import CostTracker

from aras.agents.analyst_agent import AnalystAgent
from aras.agents.citation_validator_agent import CitationValidatorAgent
from aras.agents.coder_agent import CoderAgent
from aras.agents.figures_agent import FiguresAgent
from aras.agents.github_agent import GitHubAgent
from aras.agents.memory_agent import MemoryAgent
from aras.agents.coherence_agent import CoherenceAgent
from aras.agents.novelty_agent import NoveltyAgent
from aras.agents.huggingface_agent import HuggingFaceAgent
from aras.agents.research_agent import ResearchAgent
from aras.agents.reviewer_agent import ReviewerAgent
from aras.agents.scraping_agent import ScrapingAgent
from aras.agents.writer_agent import WriterAgent

from aras.healing.health_monitor import HealthMonitor
from aras.self_improvement.prompt_evolver import PromptEvolver
from aras.self_improvement.prompt_ab_tester import PromptABTester
from aras.self_improvement.scorer import PaperScorer


log = get_logger("orchestrator")


@dataclass
class UIEvent:
    agent: str
    message: str
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {"agent": self.agent, "message": self.message, "ts": self.ts}


class Orchestrator:
    """Master agent loop coordinating planning, scraping, experiments, writing, review, and publishing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._token_lock = threading.Lock()
        self._cost_tracker = CostTracker(logs_dir=Path(settings.logs_dir).resolve())
        self._states: dict[str, AgentState] = {
            "orchestrator": AgentState(),
            "research": AgentState(),
            "novelty": AgentState(),
            "scraping": AgentState(),
            "citations": AgentState(),
            "coder": AgentState(),
            "analyst": AgentState(),
            "figures": AgentState(),
            "coherence": AgentState(),
            "writer": AgentState(),
            "reviewer": AgentState(),
            "memory": AgentState(),
            "github": AgentState(),
            "hf": AgentState(),
            "ab_tester": AgentState(),
        }
        self._pipeline: list[PipelineStep] = [
            PipelineStep("Novelty", 0.0),
            PipelineStep("Scraping", 0.0),
            PipelineStep("Citations", 0.0),
            PipelineStep("Coding", 0.0),
            PipelineStep("Experiments", 0.0),
            PipelineStep("Analysis", 0.0),
            PipelineStep("Figures", 0.0),
            PipelineStep("Writing", 0.0),
            PipelineStep("Review", 0.0),
            PipelineStep("Publish", 0.0),
            PipelineStep("Self-improve", 0.0),
        ]
        self._log_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._tokens_input_used: int = 0
        self._tokens_output_used: int = 0
        self._tokens_used: int = 0
        self._cost_usd: float = 0.0
        self._cost_per_cycle_last: float = 0.0
        self._budget_remaining_usd: float | None = None
        self._budget_exceeded: bool = False
        self._cycle_budget_exceeded: bool = False
        self._errors: int = 0
        self._topic: str | None = None
        self._cycle: int | None = None
        self._paper_preview: str = ""
        self._memory_preview: str = ""
        self._github_url: str | None = None
        self._paper_score: float | None = None
        self._current_task: str = ""

        def on_chat_result_for(agent_id: str):
            def _cb(result: Any) -> None:
                # Callback may be invoked from async contexts; keep it fast and exception-safe.
                try:
                    ti = int(getattr(result, "tokens_input"))
                    to = int(getattr(result, "tokens_output"))
                    tt = int(getattr(result, "tokens_total"))
                    provider = str(getattr(result, "provider"))
                    model = str(getattr(result, "model"))
                except Exception:
                    # If token fields are missing, best-effort to preserve the previous behavior.
                    try:
                        tt = int(getattr(result, "tokens_used", 0))
                        ti = tt
                        to = 0
                        provider = str(getattr(result, "provider", "unknown"))
                        model = str(getattr(result, "model", "unknown"))
                    except Exception:
                        return

                with self._token_lock:
                    self._tokens_input_used += ti
                    self._tokens_output_used += to
                    self._tokens_used += tt

                self._cost_tracker.record(
                    agent_id=agent_id,
                    provider=provider,
                    model=model,
                    input_tokens=ti,
                    output_tokens=to,
                )
                with self._token_lock:
                    self._cost_usd = float(self._cost_tracker.total_cost_usd)
                    self._budget_remaining_usd = self._cost_tracker.budget_remaining_usd(
                        budget_ceiling_usd=float(self.settings.budget_usd_ceiling)
                        if self.settings.budget_usd_ceiling is not None
                        else None
                    )
                    if self.settings.budget_usd_ceiling is not None and self._cost_usd >= float(self.settings.budget_usd_ceiling):
                        self._budget_exceeded = True
                        self._cycle_budget_exceeded = True

            return _cb

        self.memory = MemoryAgent(settings=settings, on_event=self._on_agent_event, on_chat_result=on_chat_result_for("memory"))
        self.research = ResearchAgent(
            settings=settings,
            memory=self.memory,
            on_event=self._on_agent_event,
            on_chat_result=on_chat_result_for("research"),
        )
        self.novelty = NoveltyAgent(
            settings=settings,
            memory=self.memory,
            on_event=self._on_agent_event,
            on_chat_result=on_chat_result_for("novelty"),
        )
        self.scraping = ScrapingAgent(settings=settings, memory=self.memory, on_event=self._on_agent_event, on_chat_result=on_chat_result_for("scraping"))
        self.coder = CoderAgent(
            settings=settings,
            memory=self.memory,
            on_event=self._on_agent_event,
            on_chat_result=on_chat_result_for("coder"),
        )
        self.citations = CitationValidatorAgent(settings=settings, on_event=self._on_agent_event)
        self.analyst = AnalystAgent(
            settings=settings,
            memory=self.memory,
            on_event=self._on_agent_event,
            on_chat_result=on_chat_result_for("analyst"),
        )
        self.figures = FiguresAgent(settings=settings, on_event=self._on_agent_event)
        self.coherence = CoherenceAgent(
            settings=settings,
            memory=self.memory,
            on_event=self._on_agent_event,
            on_chat_result=on_chat_result_for("coherence"),
        )
        self.writer = WriterAgent(
            settings=settings,
            memory=self.memory,
            on_event=self._on_agent_event,
            on_chat_result=on_chat_result_for("writer"),
        )
        self.reviewer = ReviewerAgent(
            settings=settings,
            memory=self.memory,
            on_event=self._on_agent_event,
            on_chat_result=on_chat_result_for("reviewer"),
        )
        self.github = GitHubAgent(settings=settings, memory=self.memory, on_event=self._on_agent_event)
        self.hf = HuggingFaceAgent(settings=settings, on_event=self._on_agent_event)
        self._hf_url: str | None = None

        self.scorer = PaperScorer(settings=settings, reviewer=self.reviewer)
        self.prompt_evolver = PromptEvolver(settings=settings, reviewer=self.reviewer, memory=self.memory)
        self.ab_tester = PromptABTester(
            settings=settings,
            on_event=self._on_agent_event,
            on_chat_result=on_chat_result_for("ab_tester"),
        )
        self.health = HealthMonitor(settings=settings, on_event=self._on_agent_event)
        self._shutdown = asyncio.Event()
        self._snapshot_task: asyncio.Task[None] | None = None
        # Optional human-in-the-loop approval gate.
        self.approval_gate = None

    def _on_chat_result(self, result: Any, *, agent_id: str) -> None:
        # Kept for compatibility with older patching approaches.
        _ = (result, agent_id)

    async def shutdown(self) -> None:
        """Stop background tasks."""
        self._shutdown.set()
        if self._snapshot_task:
            self._snapshot_task.cancel()
        await self.health.stop()
        await self.memory.close()

    def ui_state(self) -> dict[str, Any]:
        """Return the current UI state snapshot."""
        return {
            "topic": self._topic,
            "cycle": self._cycle,
            "agents": {k: v.to_dict() for k, v in self._states.items()},
            "pipeline": [s.to_dict() for s in self._pipeline],
            "current_task": self._current_task,
            "tokens_used": self._tokens_used,
            "tokens_input": self._tokens_input_used,
            "tokens_output": self._tokens_output_used,
            "errors": self._errors,
            "cost_usd": self._cost_usd,
            "budget_remaining_usd": self._budget_remaining_usd,
            "cost_per_cycle": self._cost_per_cycle_last,
            "paper_preview": self._paper_preview,
            "memory_preview": self._memory_preview,
            "github_url": self._github_url,
            "hf_url": self._hf_url,
            "paper_score": self._paper_score,
        }

    async def ui_logs(self) -> AsyncIterator[dict[str, Any]]:
        """Yield UI log events."""
        while not self._shutdown.is_set():
            evt = await self._log_queue.get()
            yield evt

    def _set_state(self, agent: str, *, status: str, detail: str = "") -> None:
        st = self._states[agent]
        st.status = status  # type: ignore[assignment]
        st.detail = detail
        st.last_update = datetime.now(timezone.utc).isoformat()

    def _set_step(self, name: str, progress: float) -> None:
        for s in self._pipeline:
            if s.name == name:
                s.progress = max(0.0, min(1.0, progress))
                return

    def _on_agent_event(self, agent: str, message: str, *, level: str = "info") -> None:
        if level == "error":
            self._errors += 1
        self._log_queue.put_nowait(
            {
                "agent": agent,
                "message": message,
                "level": level,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def run(self, *, topic: str, cycles: int = 1) -> FinalReport:
        """Run N research cycles for a topic."""
        t0 = time.time()
        self._topic = topic
        self._set_state("orchestrator", status="WORKING", detail="running")
        await self.health.start(agents=self._states)
        self._start_periodic_snapshots(topic=topic)

        outputs: dict[str, str] = {}
        for i in range(1, cycles + 1):
            self._cycle = i
            await self._run_cycle(topic=topic, cycle=i, outputs=outputs)

        self._set_state("orchestrator", status="DONE", detail="complete")
        return FinalReport(
            topic=topic,
            cycles=cycles,
            paper_score=self._paper_score,
            github_url=self._github_url,
            tokens_used=self._tokens_used,
            elapsed_seconds=time.time() - t0,
            outputs=outputs,
        )

    def _start_periodic_snapshots(self, *, topic: str) -> None:
        if self._snapshot_task and not self._snapshot_task.done():
            return
        interval = int(self.settings.memory_snapshot_interval_seconds)
        if interval <= 0:
            return

        async def _loop() -> None:
            while not self._shutdown.is_set():
                await asyncio.sleep(interval)
                await self.memory.snapshot(out_dir=Path(self.settings.memory_snapshot_dir).resolve(), label=_slug(topic))

        self._snapshot_task = asyncio.create_task(_loop())

    async def _run_cycle(self, *, topic: str, cycle: int, outputs: dict[str, str]) -> None:
        base = Path(".").resolve()
        paper_dir = base / "paper"
        exp_dir = base / "experiments"
        logs_dir = base / "logs"
        improvement_log = base / "IMPROVEMENT_LOG.md"
        improvement_log.parent.mkdir(parents=True, exist_ok=True)
        pivot_info: str = ""
        ab_test_note: str = ""

        self._current_task = f"cycle {cycle}: initialize memory"
        self._set_state("memory", status="WORKING", detail="loading")
        self._cycle_budget_exceeded = False
        self._cost_tracker.begin_cycle(cycle=cycle)
        await self.memory.startup()
        self._memory_preview = await self.memory.preview()
        self._set_state("memory", status="DONE", detail="ready")

        self._current_task = f"cycle {cycle}: planning research"
        self._set_state("research", status="WORKING", detail="planning")
        plan = await self.research.plan(topic=topic)
        self._set_state("research", status="DONE", detail="plan ready")

        self._current_task = f"cycle {cycle}: novelty check"
        self._set_step("Novelty", 0.05)
        self._set_state("novelty", status="WORKING", detail="cross-checking memory")
        novelty = await self.novelty.check(topic=topic, plan=plan, cycle=cycle)
        novelty_score_for_approval = float(novelty.novelty_score)
        novelty_info = (
            f"novelty={novelty.novelty_score:.3f}, conf={float(getattr(novelty, 'confidence', 0.0)):.3f}, "
            f"evidence={int(getattr(novelty, 'validated_evidence_count', 0))}/{int(getattr(novelty, 'evidence_count', 0))}, "
            f"gate={bool(getattr(novelty, 'gate_passed', False))}"
        )
        self._set_step("Novelty", 1.0)
        self._set_state(
            "novelty",
            status="DONE",
            detail=novelty_info,
        )
        self._on_agent_event("orchestrator", f"Novelty decision: {novelty_info}")

        if bool(getattr(novelty, "gate_passed", False)) and novelty.selected_angle:
            pivot_topic = f"{topic} (pivot: {novelty.selected_angle})"
            pivot_info = (
                f"- Novelty: {novelty.novelty_score:.3f}\n"
                f"- Novelty confidence: {float(getattr(novelty, 'confidence', 0.0)):.3f}\n"
                f"- Evidence: {int(getattr(novelty, 'validated_evidence_count', 0))} validated / {int(getattr(novelty, 'evidence_count', 0))} total\n"
                f"- Pivot selected: {novelty.selected_angle}\n"
            )
            self._current_task = f"cycle {cycle}: pivoting research angle"
            self._set_state("research", status="WORKING", detail="re-planning after pivot")
            plan = await self.research.plan(topic=pivot_topic)
            topic = pivot_topic
            self._set_state("research", status="DONE", detail="pivot plan ready")
        elif novelty.novelty_score < float(self.settings.novelty_pivot_max_score):
            reason = str(getattr(novelty, "gate_reason", "strict_gate_blocked"))
            self._on_agent_event("orchestrator", f"Pivot blocked by strict novelty gate: {reason}", level="warning")

        self._current_task = f"cycle {cycle}: scraping sources"
        self._set_step("Scraping", 0.05)
        self._set_state("scraping", status="WORKING", detail="scraping")
        scraped = await self.scraping.scrape(plan=plan)
        self._set_step("Scraping", 1.0)
        self._set_state("scraping", status="DONE", detail=f"{len(scraped)} items")

        self._current_task = f"cycle {cycle}: validating citations"
        self._set_step("Citations", 0.05)
        self._set_state("citations", status="WORKING", detail="crossref validation")
        scraped = await self.citations.validate(items=scraped, cycle=cycle)
        citations_for_db = [it for it in scraped if it.get("validated") and it.get("doi")]
        if citations_for_db:
            await self.memory.store_citations(citations=citations_for_db, cycle=cycle)
        self._set_step("Citations", 1.0)
        self._set_state("citations", status="DONE", detail=f"{len(citations_for_db)} stored citations")

        self._current_task = f"cycle {cycle}: coding experiments"
        self._set_step("Coding", 0.05)
        self._set_state("coder", status="WORKING", detail="generating experiments")
        exp_bundle = await self.coder.design_and_write_experiments(topic=topic, plan=plan, scraped=scraped, output_root=exp_dir)
        self._set_step("Coding", 1.0)
        self._set_state("coder", status="DONE", detail="experiments written")

        self._current_task = f"cycle {cycle}: running experiments"
        rerun_enabled = bool(self.settings.figure_quality_rerun_enabled)
        max_reruns = int(self.settings.figure_quality_max_reruns)
        max_attempts = 1 + max(0, max_reruns)
        attempt = 0
        results: dict[str, Any] = {}
        analysis: dict[str, Any] = {}

        while attempt < max_attempts:
            attempt += 1

            self._set_step("Experiments", 0.05)
            self._set_state("coder", status="WORKING", detail=f"running experiments (attempt {attempt}/{max_attempts})")
            results = await self.coder.run_experiments(exp_bundle=exp_bundle, cycle=cycle)
            self._set_step("Experiments", 1.0)
            self._set_state("coder", status="DONE", detail=f"results captured (attempt {attempt})")

            self._current_task = f"cycle {cycle}: analyzing results (attempt {attempt})"
            self._set_step("Analysis", 0.05)
            self._set_state("analyst", status="WORKING", detail="analyzing")
            analysis = await self.analyst.analyze(topic=topic, plan=plan, scraped=scraped, results=results)
            self._set_step("Analysis", 1.0)
            self._set_state("analyst", status="DONE", detail="analysis ready")

            self._current_task = f"cycle {cycle}: generating figures (attempt {attempt})"
            self._set_step("Figures", 0.05)
            self._set_state("figures", status="WORKING", detail="plotting")
            fig_updates = await self.figures.generate(topic=topic, results=results, output_dir=paper_dir)
            analysis.update(fig_updates)
            self._set_step("Figures", 1.0)
            self._set_state("figures", status="DONE", detail="figures ready")

            quality = fig_updates.get("figure_quality_summary") if isinstance(fig_updates, dict) else {}
            all_runs_degraded = bool(isinstance(quality, dict) and quality.get("all_runs_degraded"))
            if not all_runs_degraded:
                break

            can_retry = rerun_enabled and (attempt < max_attempts) and (not self._cycle_budget_exceeded)
            if not can_retry:
                self._on_agent_event(
                    "orchestrator",
                    (
                        f"Figure quality gate failed (all runs degraded) on attempt {attempt}; "
                        "continuing without rerun due to limits or budget."
                    ),
                    level="error",
                )
                break

            self._on_agent_event(
                "orchestrator",
                (
                    f"Figure quality gate failed (all runs degraded) on attempt {attempt}; "
                    f"rerunning experiments (attempt {attempt + 1}/{max_attempts})."
                ),
                level="warning",
            )
            self._set_state("orchestrator", status="WORKING", detail="rerunning degraded cycle")

        self._current_task = f"cycle {cycle}: writing paper"
        self._set_step("Writing", 0.05)
        self._set_state("writer", status="WORKING", detail="building LaTeX")
        paper_paths = await self.writer.write_paper(
            topic=topic,
            plan=plan,
            scraped=scraped,
            results=results,
            analysis=analysis,
            output_dir=paper_dir,
        )
        self._paper_preview = (paper_dir / "paper.tex").read_text(encoding="utf-8")[:12000]
        self._set_step("Writing", 1.0)
        self._set_state("writer", status="DONE", detail="paper built")

        import difflib

        diff_dir = paper_dir / "diffs"
        diff_dir.mkdir(parents=True, exist_ok=True)
        paper_tex_path = paper_paths["tex"]
        paper_tex = paper_tex_path.read_text(encoding="utf-8")

        review: dict[str, Any] = {}
        for r_round in range(1, int(self.settings.review_rounds) + 1):
            self._current_task = f"cycle {cycle}: reviewing paper (round {r_round})"
            self._set_step("Review", (r_round - 1) / max(1, int(self.settings.review_rounds)))
            self._set_state("reviewer", status="WORKING", detail=f"peer review round {r_round}")
            review = await self.reviewer.review(paper_tex_path=paper_tex_path)
            self._set_state("reviewer", status="DONE", detail=f"round {r_round} score={review.get('score', 'n/a')}")

            self._current_task = f"cycle {cycle}: coherence revision (round {r_round})"
            self._set_state("coherence", status="WORKING", detail="revising LaTeX")
            revised_tex = await self.coherence.revise(
                topic=topic,
                paper_tex=paper_tex,
                review=review,
                round=r_round,
            )

            # Compute and persist diff for UI/auditability.
            old_lines = paper_tex.splitlines(keepends=True)
            new_lines = revised_tex.splitlines(keepends=True)
            diff_lines = list(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"paper.tex@round{r_round-1}",
                    tofile=f"paper.tex@round{r_round}",
                    lineterm="",
                )
            )
            added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))

            diff_path = diff_dir / f"paper_diff_round{r_round}.patch"
            diff_path.write_text("\n".join(diff_lines) + "\n", encoding="utf-8")
            self._on_agent_event(
                "orchestrator",
                f"Saved paper diff round {r_round}: {diff_path.name} (+{added}/-{removed})",
            )

            paper_tex_path.write_text(revised_tex, encoding="utf-8")
            paper_tex = revised_tex
            self._paper_preview = paper_tex[:12000]
            self._set_step("Review", r_round / max(1, int(self.settings.review_rounds)))

            self._set_state("coherence", status="DONE", detail=f"diff +{added}/-{removed}")

            # Safety: stop if no changes were made.
            if added == 0 and removed == 0:
                break

        self._set_step("Review", 1.0)

        self._current_task = f"cycle {cycle}: scoring paper"
        self._paper_score = await self.scorer.score(paper_tex_path=paper_tex_path, review=review)

        self._current_task = f"cycle {cycle}: persisting memory + snapshots"
        await self.memory.store_cycle(topic=topic, plan=plan, scraped=scraped, results=results, analysis=analysis, review=review, paper_score=self._paper_score)
        self._memory_preview = await self.memory.preview()
        await self.memory.snapshot(out_dir=Path(self.settings.memory_snapshot_dir).resolve(), label=f"{_slug(topic)}_cycle{cycle}")

        if not self._cycle_budget_exceeded:
            self._current_task = f"cycle {cycle}: self-improving prompts"
            self._set_step("Self-improve", 0.05)
            self._set_state("orchestrator", status="WORKING", detail="evolving prompts")
            old_prompts = self.memory.current_prompts()
            evolution = await self.prompt_evolver.evolve(topic=topic, review=review)
            self._set_step("Self-improve", 1.0)
            self._set_state("orchestrator", status="WORKING", detail="prompts updated")
            new_prompts = self.memory.current_prompts()
            self._set_state("ab_tester", status="WORKING", detail="A/B testing writer prompt")
            ab = await self.ab_tester.test_writer_abstract(
                topic=topic,
                plan=plan,
                system_a=str(old_prompts.get("writer") or ""),
                system_b=str(new_prompts.get("writer") or ""),
            )
            self._set_state("ab_tester", status="DONE", detail=f"winner={ab.get('winner')}")
            ab_test_note = (
                f"- AB writer winner: {ab.get('winner')} (a={ab.get('score_a')}, b={ab.get('score_b')})\n"
            )
            evolution["ab_test"] = ab
        else:
            evolution = {"lessons_learned": "Skipped prompt evolution due to budget ceiling.", "prompt_version": None}

        entry = (
            f"## Cycle {cycle} — {datetime.now(timezone.utc).isoformat()}\n\n"
            f"- Topic: {topic}\n"
            f"- Paper score: {self._paper_score}\n"
            f"{pivot_info}"
            f"{ab_test_note}"
            f"- Lessons: {evolution.get('lessons_learned','')}\n"
            f"- Prompt version: {evolution.get('prompt_version','')}\n\n"
        )
        old = improvement_log.read_text(encoding="utf-8") if improvement_log.exists() else "# Improvement Log\n\n"
        safe_write_text(improvement_log, old + entry)

        self._current_task = f"cycle {cycle}: requesting approval to publish"
        if not self._cycle_budget_exceeded:
            approved = True
            decision_note = ""
            if self.approval_gate is not None:
                paper_tex_for_abs = paper_tex_path.read_text(encoding="utf-8")
                import re

                m = re.search(r"\\begin\\{abstract\\}([\\s\\S]*?)\\end\\{abstract\\}", paper_tex_for_abs)
                abstract = m.group(1).strip() if m else ""
                # Collapse whitespace; keep LaTeX commands as-is for webhook consumers.
                abstract = " ".join(abstract.split())[:2000]

                payload = ApprovalPayload(
                    topic=topic,
                    cycle=cycle,
                    paper_score=float(self._paper_score or 0.0),
                    novelty_score=float(novelty_score_for_approval or 0.0),
                    abstract=abstract,
                    sections_completed=[
                        "abstract",
                        "introduction",
                        "related_work",
                        "methodology",
                        "architecture",
                        "experiments",
                        "results",
                        "discussion",
                        "conclusion",
                    ],
                    github_draft_url=None,
                    cost_usd=float(self._cost_usd),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                decision = await self.approval_gate.request(payload=payload)
                approved = bool(decision.approved)
                decision_note = str(decision.note or "")

            if approved:
                self._set_step("Publish", 0.05)
                self._set_state("github", status="WORKING", detail="publishing")
                gh = await self.github.publish(topic=topic, outputs_root=base, paper_pdf=paper_paths.get("pdf"))
                self._github_url = gh.get("url")
                self._set_step("Publish", 1.0)
                self._set_state("github", status="DONE", detail=("published" if self._github_url else "skipped"))
                # Best-effort HF publish.
                try:
                    self._current_task = f"cycle {cycle}: publishing to Hugging Face"
                    self._set_state("hf", status="WORKING", detail="uploading artifacts")
                    hf_res = await self.hf.publish_dataset(
                        topic=topic,
                        outputs_root=base,
                        paper_pdf=paper_paths.get("pdf"),
                        cycle=cycle,
                    )
                    self._hf_url = hf_res.get("url")
                    self._set_state("hf", status="DONE", detail=("uploaded" if self._hf_url else "skipped"))
                except Exception as e:
                    self._set_state("hf", status="ERROR", detail=f"hf publish failed: {e}")
            else:
                self._github_url = None
                self._set_step("Publish", 1.0)
                self._set_state("github", status="DONE", detail=("skipped (approval rejected) " + decision_note)[:200])
        else:
            self._github_url = None
            self._set_step("Publish", 1.0)
            self._set_state("github", status="DONE", detail="skipped (budget ceiling)")

        outputs["paper_tex"] = str(paper_paths["tex"])
        if paper_paths.get("pdf"):
            outputs["paper_pdf"] = str(paper_paths["pdf"])
        outputs["experiments_dir"] = str(exp_dir)
        outputs["logs_dir"] = str(logs_dir)

        self._write_cycle_quality(
            logs_dir=logs_dir,
            cycle=cycle,
            topic=topic,
            novelty=novelty,
            analysis=analysis,
            review=review,
            paper_score=self._paper_score,
        )

        # Finalize cost report for the cycle (used for UI).
        report, _budget_exceeded = self._cost_tracker.end_cycle(cycle=cycle, budget_ceiling_usd=float(self.settings.budget_usd_ceiling))
        self._cost_per_cycle_last = report.total_cost_usd
        with self._token_lock:
            self._cost_usd = float(self._cost_tracker.total_cost_usd)
            self._budget_remaining_usd = self._cost_tracker.budget_remaining_usd(budget_ceiling_usd=float(self.settings.budget_usd_ceiling))

    def _write_cycle_quality(
        self,
        *,
        logs_dir: Path,
        cycle: int,
        topic: str,
        novelty: Any,
        analysis: dict[str, Any],
        review: dict[str, Any],
        paper_score: float | None,
    ) -> None:
        q = analysis.get("figure_quality_summary") if isinstance(analysis.get("figure_quality_summary"), dict) else {}
        health = q.get("health") if isinstance(q.get("health"), dict) else {}
        success_runs = int(health.get("success") or 0)
        degraded_runs = int(health.get("degraded") or 0)
        failed_runs = int(health.get("failed") or 0)
        total_runs = max(1, success_runs + degraded_runs + failed_runs)
        run_quality = float(success_runs) / float(total_runs)

        novelty_conf = _safe_float(getattr(novelty, "confidence", 0.0))
        novelty_score = _safe_float(getattr(novelty, "novelty_score", 0.5), default=0.5)
        novelty_gate = bool(getattr(novelty, "gate_passed", False))
        validated_evidence = int(getattr(novelty, "validated_evidence_count", 0) or 0)
        total_evidence = int(getattr(novelty, "evidence_count", 0) or 0)

        rev_score = _safe_float(review.get("score"), default=_safe_float(review.get("overall_score"), default=0.0))
        ps = _safe_float(paper_score, default=0.0)

        fig_high = int(q.get("high_confidence") or 0)
        fig_low = int(q.get("low_confidence") or 0)
        fig_total = max(1, fig_high + fig_low)
        fig_quality = float(fig_high) / float(fig_total)

        novelty_quality = (1.0 - novelty_score) * 0.5 + novelty_conf * 0.5
        improvement_index = max(
            0.0,
            min(10.0, 0.30 * ps + 0.25 * rev_score + 0.20 * (run_quality * 10.0) + 0.15 * (fig_quality * 10.0) + 0.10 * (novelty_quality * 10.0)),
        )

        row = {
            "cycle": int(cycle),
            "topic": str(topic),
            "paper_score": ps,
            "review_score": rev_score,
            "experiment_health": {
                "success": success_runs,
                "degraded": degraded_runs,
                "failed": failed_runs,
                "quality": run_quality,
            },
            "figure_quality": {
                "high_confidence": fig_high,
                "low_confidence": fig_low,
                "quality": fig_quality,
            },
            "novelty": {
                "score": novelty_score,
                "confidence": novelty_conf,
                "gate_passed": novelty_gate,
                "gate_reason": str(getattr(novelty, "gate_reason", "")) or None,
                "validated_evidence": validated_evidence,
                "evidence_count": total_evidence,
                "evidence_sources": list(getattr(novelty, "evidence_sources", []) or []),
            },
            "improvement_index": improvement_index,
            "cost_usd": float(self._cost_usd),
        }

        logs_dir.mkdir(parents=True, exist_ok=True)
        out = logs_dir / "cycle_quality.jsonl"
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _slug(s: str) -> str:
    import re

    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "topic"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)
