from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from aras.agents.base import BaseAgent, EventSink
from aras.agents.memory_agent import MemoryAgent
from aras.config import Settings
from aras.healing.fallback_router import FallbackRouter
from aras.types import NoveltyResult
from aras.utils.logging import get_logger


log = get_logger("novelty-agent")


class NoveltyAgent(BaseAgent):
    """Estimate novelty/oversaturation and propose a pivot angle when needed."""

    def __init__(
        self,
        settings: Settings,
        memory: MemoryAgent,
        on_event: EventSink,
        on_tokens=None,
        on_chat_result=None,
    ) -> None:
        super().__init__(agent_id="novelty", on_event=on_event, on_tokens=on_tokens, on_chat_result=on_chat_result)
        self.settings = settings
        self.memory = memory
        self.router = FallbackRouter(settings=settings)

    async def check(self, *, topic: str, plan: dict[str, Any], cycle: int) -> NoveltyResult:
        """Return novelty score and potential pivot angle."""
        evidence = await self._gather_evidence(topic=topic, plan=plan, cycle=cycle)
        heur_score, heur_conf = self._heuristic_score_from_evidence(evidence)

        llm_obj: dict[str, Any] = {}
        try:
            prompts = self.memory.current_prompts()
            rag = self.memory.rag_context(
                query=f"novelty check for {topic}",
                collections=["citation_db", "research_memory", "learned_heuristics", "experiment_results"],
            )

            evidence_preview = json.dumps(evidence[:12], ensure_ascii=False)[:12000]

            system = f"{prompts.get('novelty','')}\n\nRAG CONTEXT:\n{rag}\n\nReturn STRICT JSON."
            user = (
                "Perform a novelty/oversaturation check for the proposed research plan.\n\n"
                f"TOPIC: {topic}\n\n"
                f"HYPOTHESIS: {plan.get('hypothesis')}\n\n"
                "Score novelty_score from 0.0 (not novel, oversaturated) to 1.0 (high novelty).\n"
                "If novelty is low (<0.35), propose a selected_angle to pivot toward.\n\n"
                f"KNOWN HEURISTIC novelty_score={heur_score:.3f}, confidence={heur_conf:.3f}.\n"
                "Use those as anchors and adjust only if evidence strongly supports it.\n\n"
                "EVIDENCE (validated/normalized papers):\n"
                f"{evidence_preview}\n\n"
                "Return STRICT JSON with keys: original_topic, selected_angle, novelty_score, confidence, competing_papers, pivot_reason.\n"
                "competing_papers must be a list of objects with at least {title, year, url} (can be empty list if unknown)."
            )
            res = await self.router.chat(
                role_system=system,
                messages=[{"role": "user", "content": user}],
                purpose="novelty_check",
                prefer=["nvidia", "openai", "local"],  # reasoning mode
                thinking=True,
                temperature=0.2,
                max_tokens=900,
            )
            self.record_chat_result(res)

            obj = self._extract_json(res.text)
            if isinstance(obj, dict):
                llm_obj = obj
        except Exception as e:
            self.emit(f"Novelty LLM check failed: {e}", level="error")

        merged = self._merge_novelty(
            topic=topic,
            heur_score=heur_score,
            heur_conf=heur_conf,
            llm_obj=llm_obj,
            evidence=evidence,
        )
        return self._apply_gate(result=merged)

    def _heuristic_novelty(self, *, topic: str) -> NoveltyResult:
        """Fallback heuristic: use Chroma similarity distance as proxy for novelty."""
        try:
            hits = self.memory.store.query(collection="research_memory", text=topic, n=6)
            distances = [h.get("distance") for h in hits if h.get("distance") is not None]
            if not distances:
                score = 0.5
            else:
                # Chroma distance is typically lower when more similar.
                min_dist = min(float(d) for d in distances if d is not None)
                # Convert to [0,1], then invert (more similarity => lower novelty).
                sim = 1.0 / (1.0 + min_dist)
                score = max(0.05, min(0.95, 1.0 - sim))
            return NoveltyResult(
                original_topic=topic,
                selected_angle="",
                novelty_score=float(score),
                competing_papers=[
                    {"title": str(hit.get("metadata", {}).get("tag") or hit.get("id") or "")}
                    for hit in hits[:3]
                    if isinstance(hit, dict)
                ],
                pivot_reason=None,
                confidence=0.3,
                evidence_count=int(len(hits)),
                validated_evidence_count=0,
                evidence_sources=["research_memory"],
                gate_passed=False,
                gate_reason="heuristic_fallback_only",
            )
        except Exception:
            return NoveltyResult(
                original_topic=topic,
                selected_angle="",
                novelty_score=0.5,
                competing_papers=[],
                pivot_reason=None,
                confidence=0.2,
                evidence_count=0,
                validated_evidence_count=0,
                evidence_sources=[],
                gate_passed=False,
                gate_reason="heuristic_fallback_only",
            )

    async def _gather_evidence(self, *, topic: str, plan: dict[str, Any], cycle: int) -> list[dict[str, Any]]:
        query = self._compose_query(topic=topic, plan=plan)
        mem_docs = self.memory.store.query(collection="citation_db", text=query, n=14)
        mem_evidence = self._normalize_memory_evidence(mem_docs)

        sem_evidence = await self._semantic_scholar(query=query)
        cross_evidence = await self._crossref(query=query)
        merged = self._dedupe_evidence(mem_evidence + sem_evidence + cross_evidence)

        # Persist for auditability.
        out = Path(self.settings.logs_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"novelty_evidence_cycle{int(cycle)}.json"
        try:
            path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return merged

    def _compose_query(self, *, topic: str, plan: dict[str, Any]) -> str:
        keywords = plan.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        keys = [str(k).strip() for k in keywords if str(k).strip()][:6]
        hyp = str(plan.get("hypothesis") or "").strip()
        parts = [topic.strip(), hyp] + keys
        return " ".join([p for p in parts if p]).strip() or topic

    def _normalize_memory_evidence(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for d in docs:
            text = d.get("text")
            item: dict[str, Any] = {}
            if isinstance(text, str) and text.strip().startswith("{"):
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        item = obj
                except Exception:
                    item = {}
            if not item:
                raw_md = d.get("metadata")
                md: dict[str, Any] = raw_md if isinstance(raw_md, dict) else {}
                item = {"title": str(md.get("tag") or d.get("id") or "").strip()}

            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title and not url:
                continue
            out.append(
                {
                    "title": title or url,
                    "year": str(item.get("published") or item.get("year") or ""),
                    "url": url,
                    "doi": str(item.get("doi") or "").lower().strip() or None,
                    "source": str(item.get("source") or "citation_db"),
                    "validated": bool(item.get("validated", True)),
                    "citation_count": int(item.get("citation_count") or 0),
                }
            )
        return out

    async def _semantic_scholar(self, *, query: str) -> list[dict[str, Any]]:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": 8,
            "fields": "title,year,url,citationCount,externalIds",
        }
        try:
            async with httpx.AsyncClient(timeout=25) as client:
                r = await client.get(url, params=params, headers={"User-Agent": "ARAS/1.0"})
                r.raise_for_status()
                payload = r.json()
        except Exception as e:
            self.emit(f"Novelty Semantic Scholar fetch failed: {e}", level="error")
            return []

        out: list[dict[str, Any]] = []
        for rec in payload.get("data") or []:
            if not isinstance(rec, dict):
                continue
            title = str(rec.get("title") or "").strip()
            url_p = str(rec.get("url") or "").strip()
            if not title and not url_p:
                continue
            raw_ext = rec.get("externalIds")
            ext: dict[str, Any] = raw_ext if isinstance(raw_ext, dict) else {}
            out.append(
                {
                    "title": title or url_p,
                    "year": str(rec.get("year") or ""),
                    "url": url_p,
                    "doi": str(ext.get("DOI") or "").lower().strip() or None,
                    "source": "semantic_scholar",
                    "validated": False,
                    "citation_count": int(rec.get("citationCount") or 0),
                }
            )
        return out

    async def _crossref(self, *, query: str) -> list[dict[str, Any]]:
        url = "https://api.crossref.org/works"
        params = {"query.bibliographic": query, "rows": "8"}
        if self.settings.crossref_email:
            params["mailto"] = self.settings.crossref_email
        try:
            async with httpx.AsyncClient(timeout=25) as client:
                r = await client.get(url, params=params, headers={"User-Agent": "ARAS/1.0"})
                r.raise_for_status()
                payload = r.json()
        except Exception as e:
            self.emit(f"Novelty Crossref fetch failed: {e}", level="error")
            return []

        out: list[dict[str, Any]] = []
        raw_message = payload.get("message") if isinstance(payload, dict) else {}
        message: dict[str, Any] = raw_message if isinstance(raw_message, dict) else {}
        for work in message.get("items") or []:
            if not isinstance(work, dict):
                continue
            t = work.get("title") or []
            title = str(t[0]).strip() if isinstance(t, list) and t else ""
            doi = str(work.get("DOI") or "").strip().lower()
            url_p = str(work.get("URL") or (f"https://doi.org/{doi}" if doi else "")).strip()
            year = ""
            issued = work.get("issued") if isinstance(work.get("issued"), dict) else {}
            parts = issued.get("date-parts") if isinstance(issued, dict) else []
            if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
                year = str(parts[0][0])
            if not title and not url_p:
                continue
            out.append(
                {
                    "title": title or url_p,
                    "year": year,
                    "url": url_p,
                    "doi": doi or None,
                    "source": "crossref",
                    "validated": True,
                    "citation_count": int(work.get("is-referenced-by-count") or 0),
                }
            )
        return out

    def _dedupe_evidence(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_key: dict[str, dict[str, Any]] = {}
        for it in items:
            title = str(it.get("title") or "").strip()
            doi = str(it.get("doi") or "").strip().lower()
            url = str(it.get("url") or "").strip().lower()
            if doi:
                key = f"doi:{doi}"
            elif title:
                key = f"title:{title.lower()}"
            elif url:
                key = f"url:{url}"
            else:
                continue

            prev = by_key.get(key)
            if prev is None:
                by_key[key] = dict(it)
                continue

            merged = dict(prev)
            merged["validated"] = bool(prev.get("validated")) or bool(it.get("validated"))
            merged["citation_count"] = max(int(prev.get("citation_count") or 0), int(it.get("citation_count") or 0))
            if not merged.get("url") and it.get("url"):
                merged["url"] = it.get("url")
            # Track source union as comma list for transparency.
            srcs = {str(prev.get("source") or ""), str(it.get("source") or "")}
            merged["source"] = ",".join(sorted([s for s in srcs if s]))
            by_key[key] = merged

        out = list(by_key.values())
        out.sort(key=lambda x: (bool(x.get("validated")), int(x.get("citation_count") or 0)), reverse=True)
        return out

    def _heuristic_score_from_evidence(self, evidence: list[dict[str, Any]]) -> tuple[float, float]:
        n = len(evidence)
        if n == 0:
            return 0.8, 0.2
        validated = sum(1 for e in evidence if bool(e.get("validated")))
        sources: set[str] = set()
        for e in evidence:
            raw = str(e.get("source") or "")
            for s in raw.split(","):
                if s.strip():
                    sources.add(s.strip())

        # More validated evidence implies higher saturation => lower novelty.
        sat = min(1.0, (validated / 8.0) + (max(0, n - 8) / 20.0))
        novelty = max(0.05, min(0.95, 1.0 - sat))
        confidence = max(0.1, min(0.95, (validated / max(1, n)) * 0.7 + min(1.0, len(sources) / 3.0) * 0.3))
        return novelty, confidence

    def _merge_novelty(
        self,
        *,
        topic: str,
        heur_score: float,
        heur_conf: float,
        llm_obj: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> NoveltyResult:
        score = self._safe_float(llm_obj.get("novelty_score"), default=heur_score)
        confidence = self._safe_float(llm_obj.get("confidence"), default=heur_conf)
        # Bound LLM drift around heuristic unless confidence is very high.
        if confidence < 0.8:
            lower = max(0.0, heur_score - 0.2)
            upper = min(1.0, heur_score + 0.2)
            score = max(lower, min(upper, score))

        llm_comp_raw = llm_obj.get("competing_papers")
        llm_comp: list[Any] = llm_comp_raw if isinstance(llm_comp_raw, list) else []
        normalized_llm = self._normalize_llm_competing(llm_comp)
        merged_competing = self._dedupe_evidence(evidence + normalized_llm)

        srcs: set[str] = set()
        validated = 0
        for e in merged_competing:
            if bool(e.get("validated")):
                validated += 1
            raw = str(e.get("source") or "")
            for s in raw.split(","):
                if s.strip():
                    srcs.add(s.strip())

        return NoveltyResult(
            original_topic=str(llm_obj.get("original_topic") or topic),
            selected_angle=str(llm_obj.get("selected_angle") or "").strip(),
            novelty_score=max(0.0, min(1.0, float(score))),
            competing_papers=merged_competing[:20],
            pivot_reason=str(llm_obj.get("pivot_reason") or "") or None,
            confidence=max(0.0, min(1.0, float(confidence))),
            evidence_count=len(merged_competing),
            validated_evidence_count=int(validated),
            evidence_sources=sorted(srcs),
            gate_passed=False,
            gate_reason=None,
        )

    def _normalize_llm_competing(self, items: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            title = str(it.get("title") or "").strip()
            url = str(it.get("url") or "").strip()
            if not title and not url:
                continue
            out.append(
                {
                    "title": title or url,
                    "year": str(it.get("year") or ""),
                    "url": url,
                    "doi": str(it.get("doi") or "").lower().strip() or None,
                    "source": "llm",
                    "validated": False,
                    "citation_count": int(it.get("citation_count") or 0),
                }
            )
        return out

    def _apply_gate(self, *, result: NoveltyResult) -> NoveltyResult:
        max_score = float(self.settings.novelty_pivot_max_score)
        min_conf = float(self.settings.novelty_min_confidence)
        min_valid = int(self.settings.novelty_min_validated_evidence)
        min_sources = int(self.settings.novelty_min_evidence_sources)
        source_count = len(result.evidence_sources)

        if result.novelty_score >= max_score:
            return result.model_copy(update={"selected_angle": "", "gate_passed": False, "gate_reason": "novelty_not_low_enough"})
        if result.confidence < min_conf:
            return result.model_copy(update={"selected_angle": "", "gate_passed": False, "gate_reason": "insufficient_confidence"})
        if int(result.validated_evidence_count) < min_valid:
            return result.model_copy(update={"selected_angle": "", "gate_passed": False, "gate_reason": "insufficient_validated_evidence"})
        if source_count < min_sources:
            return result.model_copy(update={"selected_angle": "", "gate_passed": False, "gate_reason": "insufficient_source_diversity"})
        if not str(result.selected_angle).strip():
            return result.model_copy(update={"gate_passed": False, "gate_reason": "no_selected_angle"})
        return result.model_copy(update={"gate_passed": True, "gate_reason": "strict_evidence_gate_passed"})

    def _safe_float(self, x: Any, *, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _extract_json(self, text: str) -> Any:
        import json
        import re

        t = text.strip()
        if t.startswith("{") and t.endswith("}"):
            return json.loads(t)
        m = re.search(r"\{[\s\S]+\}", t)
        if not m:
            raise ValueError("no JSON found")
        return json.loads(m.group(0))
