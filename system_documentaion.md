# ARAS System Documentation

This document describes the **Autonomous Research Agent System (ARAS)** that was implemented in this workspace, including architecture, runtime behavior, configuration, file structure, agent responsibilities, self-healing/self-improvement, storage formats, UI/WebSocket schema, Docker, and operational runbooks.

> Note on “mission completeness”: the system is **runnable end-to-end** with graceful fallbacks when API keys or local models are missing. When keys/models are available, ARAS upgrades behavior automatically through its provider fallback router.

## Architecture overview

ARAS is a **single-process orchestrator** (asyncio) that coordinates a hierarchy of agents. It maintains:
- **Live state** (agent statuses, pipeline progress, current task, counters)
- **Persistent memory** (ChromaDB, on-disk) + prompt versions
- **A research cycle pipeline** that produces:
  - scraped evidence
  - runnable experiment code + artifacts (figures + `results.json`)
  - analysis tables/narrative
  - LaTeX paper (`paper/paper.tex`) and optionally PDF (`paper/paper.pdf`)
  - an improvement log entry (`IMPROVEMENT_LOG.md`)
  - optional GitHub repository publishing and release

Provider routing and resilience:
- **Local OpenAI-compatible servers** (llama-cpp-python) if configured/running
- **NVIDIA integrate API** (SSE streaming, “thinking” supported)
- **OpenAI chat completions** (standard endpoint)
- **Exponential backoff retries** per provider, then fallback to the next provider

Real-time UI:
- **FastAPI** + **WebSockets** emits state snapshots and log events
- A **single-file vanilla JS UI** renders agent board, logs, progress pipeline, paper preview, memory preview, and GitHub URL.

## Implemented file tree (high-level)

Top-level additions:
- `aras/` — main package + runtime code
- `IMPROVEMENT_LOG.md` — appended each cycle with score/lessons/prompt version
- `system_documentaion.md` — this document

Primary ARAS package structure:

```text
aras/
  main.py
  orchestrator.py
  config.py
  types.py
  requirements.txt
  README.md
  .env.example
  Dockerfile
  docker-compose.yml
  quick_start.sh
  quick_start.ps1
  agents/
  models/
  memory/
  scraping/
  experiments/
  paper/
  ui/
  self_improvement/
  healing/
  utils/
```

## Entry point and runtime

### `aras/main.py`
**Purpose**: CLI entry point.

Behavior:
- Parses CLI args:
  - `--topic` (required)
  - `--cycle` (default 1)
  - `--no-browser`
- Ensures directories exist via `Settings.resolved_paths()`
- Configures logging to:
  - console (Rich)
  - file (`logs/aras.log`)
- Starts UI server (FastAPI/uvicorn) as an asyncio task
- Starts orchestrator research run:
  - `await orchestrator.run(topic=..., cycles=...)`
- Prints final report JSON to logs
- Shuts down health monitor + memory

### `aras/orchestrator.py`
**Purpose**: master pipeline coordinator.

In-memory state tracked for UI:
- `agents`: map of agent id → `{status, last_update, detail}`
- `pipeline`: progress steps:
  - Novelty → Scraping → Citations → Coding → Experiments → Analysis → Figures → Writing → Review → Publish → Self-improve
- current task string
- counters and dashboards:
  - `tokens_used`, `tokens_input`, `tokens_output`
  - `errors`
  - `cost_usd`, `budget_remaining_usd`, `cost_per_cycle`
- previews: `paper_preview`, `memory_preview`
- publish outputs: `github_url`, `hf_url`, `paper_score`

Cycle flow (per `Orchestrator._run_cycle`):
1. **Memory init**: `MemoryAgent.startup()` + preview
2. **Planning**: `ResearchAgent.plan(topic)`
3. **Novelty check + strict evidence-gated pivoting**: `NoveltyAgent.check(...)`
   - novelty score is computed from normalized evidence + bounded LLM refinement
   - pivoting is only allowed when strict gate passes (`gate_passed=true`)
   - if gate fails, orchestrator logs `gate_reason` and keeps the original topic
4. **Scraping**: `ScrapingAgent.scrape(plan)` → list of structured items
5. **Citation validation + enrichment**: `CitationValidatorAgent.validate(...)` using Crossref
6. **Experiment codegen**: `CoderAgent.design_and_write_experiments(...)` (domain-parameterized experiments + metric streaming)
7. **Experiment execution**: `CoderAgent.run_experiments(...)` → live metric events + `results.json`
8. **Analysis**: `AnalystAgent.analyze(...)` → markdown table + narrative
9. **Figures generation**: `FiguresAgent.generate(...)` (quality-gated plots + TikZ snippet)
   - low-confidence figures are generated for debugging/UI but excluded from paper LaTeX
   - if all runs are degraded, orchestrator reruns experiments/analysis/figures up to configured limits
10. **Writing**: `WriterAgent.write_paper(...)` → `paper/paper.tex` + bib + optional PDF
11. **Review + coherence revisions**: `ReviewerAgent.review(...)` + `CoherenceAgent.revise(...)` with unified diffs
12. **Scoring**: `PaperScorer.score(...)`
13. **Persist memory + snapshots**: `MemoryAgent.store_cycle(...)` + local Chroma zip snapshots
14. **Self-improve prompts + A/B test**: `PromptEvolver.evolve(...)` + `PromptABTester`
15. **Publish (gated + best-effort)**: approval webhook → `GitHubAgent.publish(...)` and `HuggingFaceAgent.publish_dataset(...)`
16. **Cycle quality ledger**: orchestrator appends objective quality metrics to `logs/cycle_quality.jsonl`

## Configuration and environment variables

### `aras/config.py`
Settings are loaded via `pydantic-settings` from **environment variables** plus a local `.env` file.

**.env discovery (important)**:
- ARAS will look for **`aras/.env` first**, then fall back to a repo-root **`.env`**.
- This makes local usage on Windows less error-prone when you keep runtime secrets next to the `aras/` package.

Key environment variables (see `aras/.env.example`):
- **Model APIs**
  - `OPENAI_API_KEY`
  - `NVIDIA_API_KEY`
- **Citation validation**
  - `CROSSREF_EMAIL` (optional; used as Crossref contact)
- **Hugging Face publishing**
  - `HF_TOKEN`
  - `HF_USERNAME`
- **Human approval gate**
  - `APPROVAL_WEBHOOK_URL`
  - `APPROVAL_TIMEOUT_SECONDS`
- **Budget / cost enforcement**
  - `BUDGET_USD_CEILING`
- **Figure quality rerun gate**
  - `FIGURE_QUALITY_RERUN_ENABLED`
  - `FIGURE_QUALITY_MAX_RERUNS`
- **Strict novelty pivot gate**
  - `NOVELTY_PIVOT_MAX_SCORE` (default `0.35`)
  - `NOVELTY_MIN_CONFIDENCE` (default `0.65`)
  - `NOVELTY_MIN_VALIDATED_EVIDENCE` (default `3`)
  - `NOVELTY_MIN_EVIDENCE_SOURCES` (default `2`)
- **GitHub publishing**
  - `GITHUB_TOKEN`
  - `GITHUB_OWNER` (optional; org/user selection)
- **Local model servers**
  - `LOCAL_MODEL_1_PATH`, `LOCAL_MODEL_2_PATH`, `LOCAL_MODEL_3_PATH`
  - `LOCAL_MODEL_HOST`
  - `LOCAL_MODEL_PORT_BASE` (ports are base, base+1, base+2)
- **Persistence / infra**
  - `CHROMA_PERSIST_DIR` (default `./chroma_db`)
  - `REDIS_URL` (default `redis://localhost:6379`)
  - `MEMORY_SNAPSHOT_INTERVAL_SECONDS` (default `900`) — periodic local zip snapshots of `chroma_db/`

Important defaults:
- NVIDIA default model: `moonshotai/kimi-k2.5`
- OpenAI default model: `gpt-4.1` (can be changed in env by editing code or env mapping)
- Escalation model for repeated coder failures: `gpt-4o`

### Secrets hygiene (read before deployment)
- **Never commit `.env`**. Keep it local only.
- If you ever paste keys into logs/screenshots/chat, **rotate them** (OpenAI/NVIDIA/GitHub/HF tokens).

## Provider routing, retries, and fallbacks (self-healing)

### `aras/healing/fallback_router.py`
**Purpose**: standardized chat completion calls across providers with:
- **Retries**: tenacity + exponential jitter backoff
- **Fallback**: tries providers in a preferred order and moves on when one fails

Providers:
- `local`: `LocalModelClient` (OpenAI-compatible llama-cpp server)
- `nvidia`: `NvidiaClient` (SSE streaming; supports “thinking”)
- `openai`: `OpenAIClient` (standard OpenAI endpoint)

API:
- `FallbackRouter.chat(...) -> ChatResult(text, provider, model, tokens_used)`

### `aras/healing/health_monitor.py`
**Purpose**: lightweight heartbeat + best-effort local server restart.

Behavior:
- runs every `Settings.heartbeat_interval_seconds` (default 30s)
- emits a “heartbeat” event
- calls `LocalModelServerManager.start_all()` to ensure local servers are up (best-effort)

### Fault tolerance behavior implemented
- **Local model missing/unreachable**: local chat fails → router falls back to NVIDIA/OpenAI based on preference.
- **NVIDIA/OpenAI key missing**: client raises → router falls back.
- **General failures**: retried with exponential backoff; if exhausted, next provider tried.
- **Experiment crashes**:
  - Coder auto-retries up to 3 times
  - attempts LLM-generated unified diff patching (restricted to valid 1-file diffs)
  - if patching fails, performs targeted simplification based on a structured failure taxonomy
  - classified failures are persisted to Chroma collection `failure_db`
- **Soft budget enforcement**: once `BUDGET_USD_CEILING` is reached, optional steps are skipped for the cycle.
- **Figure quality reruns**: when figures report `all_runs_degraded=true`, orchestrator retries experiments/analysis/figures (bounded by settings + budget).
- **Strict novelty anti-false-pivot gate**: novelty pivots are blocked unless evidence, confidence, and source-diversity thresholds all pass.
- **Publish failures**: missing tokens or approval rejection skips GitHub/HF publishing; paper generation remains local.

## Agents (responsibilities + main methods)

All agents inherit from:

### `aras/agents/base.py`
- **`BaseAgent.emit(message, level=...)`**: sends UI log events via `on_event` sink
- **`BaseAgent.guarded(kind, fn)`**: structured try/except wrapper producing a structured error dict

### `aras/agents/memory_agent.py` (Persistent Memory Agent)
Backed by ChromaDB (`chromadb.PersistentClient`).

Collections created/used:
- `research_memory`
- `agent_feedback` (created; not yet populated heavily)
- `learned_heuristics`
- `experiment_results`
- `citation_db`
- `failure_db`
- `prompt_versions`

Key methods:
- `startup()`: ensures collections and prompt version exist; stores latest prompt version into `prompt_versions`
- `rag_context(query, collections=None)`: RAG snippets across selected collections; novelty path can explicitly include `citation_db`
- `store_cycle(...)`: stores cycle summary + heuristics + experiment results into collections
- `store_citations(...)`: persists validated Crossref-enriched citation metadata into `citation_db`
- `store_failure(...)`: persists structured experiment failure taxonomy into `failure_db`
- `bump_prompts(updated)`: writes `prompt_versions/prompts_vN.json` and persists prompt version into `prompt_versions` collection
- `preview()`: returns a compact memory preview for the UI

### `aras/agents/research_agent.py` (Research/Planning Agent)
- `plan(topic) -> dict`: tries NVIDIA/OpenAI/local to produce STRICT JSON plan:
  - `hypothesis`, `questions`, `experiments`, `metrics`, `outline`, `keywords`
- If LLMs unavailable: deterministic fallback plan is returned.

### `aras/agents/scraping_agent.py` (Scraping Agent)
Implements “scrape sources” step:
- Builds a query from `plan["keywords"]`
- Pulls:
  - arXiv entries via the arXiv API (`export.arxiv.org/api/query`)
  - Wikipedia search page
  - GitHub search page
  - PapersWithCode search page
  - Google Scholar search page (best-effort; may be blocked)
  - Semantic Scholar Graph API items (deep API tier)
  - PapersWithCode API items (deep API tier)
- Uses `ScrapeRouter.fetch()` to get HTML (with cache)
- Parses via:
  - `parse_arxiv` (`aras/scraping/parsers/arxiv_parser.py`)
  - `parse_github` (`aras/scraping/parsers/github_parser.py`)
  - `parse_generic` (`aras/scraping/parsers/generic_parser.py`)
  - `parse_paperswithcode` (`aras/scraping/parsers/paperswithcode_parser.py`)
  - `parse_google_scholar` (`aras/scraping/parsers/scholar_parser.py`)
- Adds a simple keyword overlap boost to `relevance` and returns ranked items.

#### Scrapling compatibility note
The upstream `scrapling` `Fetcher` constructor has changed across versions. ARAS initializes it in a
backwards-compatible way (tries `Fetcher(stealthy=True)` then falls back to `Fetcher()` if unsupported).

### `aras/agents/coder_agent.py` (Experiment Agent)
Creates and runs experiments under `experiments/{paper_slug}/`.

Key methods:
- `design_and_write_experiments(...) -> ExperimentBundle`
  - infers a coarse domain (`nlp`/`cv`/`rl`/`general`) from plan/topic keywords
  - writes deterministic, CPU-only synthetic experiments with domain-parameterized hyperparameters:
    - `exp1_synthetic_linear.py`
    - `exp2_ablation_lr.py`
    - `exp3_robustness_noise.py`
  - writes `manifest.json`
  - writes unit test `tests/test_experiments_import.py`
- `run_experiments(bundle) -> dict`
  - copies each module into its own run directory and executes it in an isolated subprocess
  - captures stdout/stderr, wall time, peak RSS, and artifacts
  - streams live metrics by parsing stdout lines prefixed with `METRIC_JSON`
  - writes aggregated `results.json` at the bundle root
- Auto-debug/simplification:
  - up to 3 attempts per experiment
  - escalation routing preference to OpenAI `gpt-4o` after repeated failures (if key present)
  - attempts LLM-generated unified diff patching (restricted validation) and applies it when safe
  - if patching fails, classifies the failure and performs targeted simplification based on the failure type

### `aras/agents/citation_validator_agent.py` (Crossref Citation Validator)
- Resolves scraped DOI or (fallback) bibliographic title against Crossref REST API
- Enriches scraped items with canonical title/authors/year/url
- Detects potential retraction via Crossref relation heuristics
- Writes validated/enriched items into Chroma collection `citation_db`

### `aras/agents/novelty_agent.py` (Novelty Check + Pivoting)
- Performs evidence-backed novelty estimation and pivot recommendation.
- Evidence sources:
  - memory `citation_db` retrieval
  - Semantic Scholar API
  - Crossref API
- Normalizes/deduplicates competing papers and persists per-cycle evidence:
  - `logs/novelty_evidence_cycle{N}.json`
- Returns enriched novelty payload with:
  - `novelty_score`, `confidence`
  - `evidence_count`, `validated_evidence_count`, `evidence_sources`
  - `gate_passed`, `gate_reason`
- Applies strict evidence gate before allowing pivot.

### `aras/agents/figures_agent.py` (Publication Figures)
- Generates publication-quality matplotlib/seaborn plots from experiment outputs.
- Uses data quality gating to avoid bogus visuals:
  - excludes failed runs from core performance/curve figures
  - rejects invalid single-point/placeholder curve data
  - emits a diagnostic figure when no valid multi-step series exists
- Produces structured figure metadata:
  - `figure_inventory`
  - `paper_eligible_figures`
  - `figure_quality_summary`
- Only high-confidence figures are included in `figures_latex` for the paper.
- Copies native experiment artifacts (`loss.png`) into `paper/figures/experiments/` when valid.
- Writes structured caption metadata to `paper/figures/captions.json`.
- Emits a reusable TikZ architecture diagram snippet for the paper.

### `aras/agents/coherence_agent.py` (Coherence + LaTeX Revision)
- Revises the full LaTeX document using reviewer feedback + required revisions
- Produces complete `revised_tex` which the orchestrator diffs and writes back
- Coherence + review repeats for `Settings.review_rounds`, with unified diffs persisted to `paper/diffs/`

### `aras/self_improvement/prompt_ab_tester.py` (Prompt A/B Tester)
- Compares old vs evolved `writer` prompt after `PromptEvolver` runs
- Runs a small abstract-writing microtask twice and picks a winner via heuristic scoring

### `aras/approval/gate.py` (Human Approval Gate)
- If `APPROVAL_WEBHOOK_URL` is configured, waits for webhook `ApprovalDecision` before publishing
- If not configured, publishing proceeds automatically (so the pipeline stays runnable)

### `aras/agents/huggingface_agent.py` (Hugging Face Publishing)
- Best-effort dataset repo creation/upload (does not block core pipeline)
- Uploads artifacts such as `experiments/`, `paper/`, `logs/`, and available memory snapshots

### `aras/agents/analyst_agent.py` (Results Analyst)
- Builds a Pandas DataFrame from `results["runs"]`
- Produces:
  - `table_markdown`
  - `narrative` (via LLM when available, else heuristic)
  - `lessons_learned`
- Writes a log artifact: `logs/analysis.json`

### `aras/agents/writer_agent.py` (Paper Writer)
Builds an IEEE-style LaTeX paper in `paper/`.

- Drafts sections (LLM when available; else fallbacks):
  - abstract, introduction, related_work, methodology, architecture, experiments, results, discussion, conclusion
- Uses:
  - `aras/paper/templates/ieee_template.tex` (Jinja2)
  - `aras/paper/bibliography.py` to auto-generate `references.bib` from scraped items
  - `aras/paper/latex_builder.py` to render `paper.tex` and compile `paper.pdf` if TeX engine exists

Compilation:
- Uses `tectonic` if present, else `pdflatex`.
- If no TeX engine exists, PDF compilation is skipped; LaTeX source remains.

### `aras/agents/reviewer_agent.py` (Peer Reviewer)
- Reviews `paper/paper.tex` and returns STRICT JSON:
  - `overall_score` and component scores
  - major/minor issues
  - required revisions
  - lessons learned
- Uses “thinking” mode when calling NVIDIA (when available).
- Fallback review returns a reasonable default score and revision list.

### `aras/agents/github_agent.py` (Publisher)
Publishes outputs to GitHub using **PyGithub** and **GitPython** (best-effort).

Behavior:
- Requires `GITHUB_TOKEN`; if missing, publishing is skipped.
- Publishing also requires approval when `APPROVAL_WEBHOOK_URL` is configured (orchestrator + `ApprovalGate`).
- Creates a new repo named:
  - `autonomous-research-{topic_slug}-{timestamp}`
- Clones it to a temporary local folder: `_publish_tmp/`
- Copies the following if present:
  - `experiments/`
  - `paper/`
  - `logs/`
  - `IMPROVEMENT_LOG.md`
  - `chroma_db/`
- Creates `memory_snapshot/chroma_db.zip` (compressed snapshot of `chroma_db/`) when publishing
- Generates `README.md` and pushes a commit
- Creates a GitHub Release and uploads `paper/paper.pdf` when available
- Opens a “Research Summary” GitHub issue
- Applies repo topics (best-effort) based on topic slug tokens

Returned publish object:
- `{ "url": <repo url or null>, "repo": <owner/name> }`

## Models layer (LLM clients + local servers)

### `aras/models/openai_client.py`
Minimal async OpenAI client:
- Endpoint: `https://api.openai.com/v1/chat/completions`
- Requires `OPENAI_API_KEY`
- Returns `(text, total_tokens_used)` when API provides usage.

### `aras/models/nvidia_client.py`
NVIDIA integrate API client implementing the required SSE pattern:
- Endpoint: `https://integrate.api.nvidia.com/v1/chat/completions`
- Header: `Accept: text/event-stream`
- Payload fields:
  - `stream: true`
  - `chat_template_kwargs: {"thinking": <bool>}`
- Parses `data:` SSE lines until `[DONE]`, assembling `delta.content`

Requires `NVIDIA_API_KEY`.

### `aras/models/local_model_server.py`
Two components:

- **`LocalModelServerManager`**
  - Best-effort starts up to 3 `llama_cpp.server` processes (OpenAI-compatible).
  - Uses `sys.executable -m llama_cpp.server --model ... --host ... --port ...`
  - Skips startup if the model file is missing or the port is already open.

- **`LocalModelClient`**
  - Calls `http://{LOCAL_MODEL_HOST}:{port}/v1/chat/completions`
  - Maps model ids to ports:
    - `local` / `local-coder` → base port
    - `local-analyst` → base+1
    - `local-writer` → base+2

## Scraping subsystem

### `aras/scraping/cache.py`
**Redis-backed HTML cache**:
- Key: `sha256(url)` with prefix `scrape:`
- TTL: `Settings.scrape_cache_ttl_seconds` (default 24h)
- If Redis is unavailable, failures are logged and scraping continues without cache.

### `aras/scraping/scrape_router.py`
Fetch pipeline:
- First checks Redis cache.
- Primary fetch: **Scrapling** `Fetcher(stealthy=True)` (sync) executed in a thread.
- JS-heavy routing: **PlaywrightFetcher** (Chromium, headless) for hosts/paths likely requiring JS (e.g., search pages).
- Fallback fetch: `httpx` GET with `User-Agent: ARAS/1.0`
- Returns `ScrapedPage { url, html, fetched_via }`

### Parsers (`aras/scraping/parsers/`)
Structured extraction of `{title, abstract, authors, published, relevance, ...}`:
- `arxiv_parser.py`: extracts title/abstract/authors/date from arXiv abs pages
- `github_parser.py`: extracts repo name + description + README text snippet (best-effort)
- `generic_parser.py`: heuristic title + meta description / first paragraph
- `paperswithcode_parser.py`: extracts paper/task metadata from PapersWithCode pages (best-effort)
- `scholar_parser.py`: extracts paper results/snippets from Google Scholar HTML (best-effort)

## Memory subsystem (persistent RAG + prompt versions)

### `aras/memory/vector_store.py`
ChromaDB wrapper (`chromadb.PersistentClient`) that:
- creates named collections on-demand
- supports:
  - `upsert(collection, docs)`
  - `query(collection, text, n)`

### `aras/memory/rag.py`
RAG context builder:
- queries multiple collections and produces a compact, prompt-safe context string
- truncates long texts to keep prompts bounded

### `aras/memory/prompt_manager.py`
Prompt versioning:
- Default prompts are embedded in `DEFAULT_PROMPTS`
- Stored in `prompt_versions/prompts_v{N}.json`
- `latest()` creates `prompts_v1.json` automatically if none exist
- `bump(updated_prompts=...)` increments the version and writes the file

Prompt versions are also stored in ChromaDB collection `prompt_versions`.

### Local snapshots (no GitHub required)
The orchestrator triggers:
- **Per-cycle snapshot**: after `store_cycle(...)`, a zip snapshot is written to `memory_snapshot/` (independent of GitHub publishing)
- **Periodic snapshot loop**: every `MEMORY_SNAPSHOT_INTERVAL_SECONDS`
- This preserves local ChromaDB state even when publishing is skipped due to missing tokens, approval rejection, or budget limits.

## Self-improvement subsystem

### `aras/self_improvement/prompt_evolver.py`
After each cycle, ARAS evolves prompts:
- Inputs:
  - current prompts
  - reviewer feedback JSON
  - memory RAG context
- Tries LLM-based prompt engineering first (NVIDIA/OpenAI/local).
- On failure, applies heuristic updates (e.g., append major issues guidance).
- Persists:
  - `prompt_versions/prompts_v{N}.json`
  - ChromaDB `prompt_versions` doc

Returns:
- `{ prompt_version: N, lessons_learned: "..." }`

### `aras/self_improvement/scorer.py`
Computes scalar paper score:
- prefers `overall_score` or `score`
- otherwise averages `novelty/methodology/clarity/reproducibility`

### `aras/self_improvement/prompt_ab_tester.py`
Runs a simple A/B test after prompt evolution:
- compares old vs evolved `writer` prompt on a micro abstract-writing task
- selects a winner via heuristic scoring

### `aras/self_improvement/failure_taxonomy.py`
Classifies experiment crashes into stable categories and supports targeted healing:
- used by `CoderAgent` to choose failure-type-specific simplification hyperparameters
- structured failures are persisted into Chroma collection `failure_db`

## Paper subsystem

### `aras/paper/templates/ieee_template.tex`
IEEETran conference template with placeholders for:
- abstract, keywords
- introduction, related work, methodology, architecture, experiments, results, discussion, conclusion
- BibTeX references using `IEEEtran` style

### `aras/paper/bibliography.py`
Bibliography generation from scraped items:
- Produces `references.bib` containing `@misc` entries derived from:
  - `title`, `authors`, `published` (year extracted), `url`, and source note
- Caps entries (default 25–30)
- Provides helper to emit `\\cite{key}` strings for related work

### `aras/paper/latex_builder.py`
Paper builder:
- Uses Jinja2 template rendering
- Writes:
  - `paper/paper.tex`
  - `paper/references.bib`
- Compiles `paper.pdf` if a TeX engine is present:
  - `tectonic` preferred
  - else `pdflatex`

## Experiments subsystem

### `aras/experiments/runner.py`
Isolated subprocess experiment runner:
- Executes `sys.executable <module.py>` inside a dedicated workdir
- Captures:
  - exit code
  - wall time
  - peak RSS (via `psutil`)
  - stdout/stderr (tail-capped)
- Discovers artifacts in the workdir (`*.png`, `*.pdf`, `*.json`)
- Optionally parses stdout lines prefixed with `METRIC_JSON` and forwards metric events for live UI logs

Experiment output convention:
- Each experiment writes a local `results.json` with metrics, and saves `loss.png` + `loss.pdf`.
- Live metric streaming:
  - experiments print lines like `METRIC_JSON { ... }` to stdout during training
  - `CoderAgent` converts those to UI log events in real time

## UI subsystem (real-time state)

### `aras/ui/server.py`
FastAPI server:
- `GET /` serves `aras/ui/static/index.html`
- `GET /api/diffs` lists revision rounds and metadata
- `GET /api/figures` lists current PNG figures with captions (supports string and structured caption objects)
- `GET /api/hf` returns latest Hugging Face publish URLs (best-effort)
- `GET /api/quality` returns latest objective cycle-quality row from `logs/cycle_quality.jsonl`
- `WS /ws/status`:
  - accepts connection
  - broadcasts a full **state snapshot** every 1 second

Background log pump:
- `orchestrator.ui_logs()` yields log events and server broadcasts them as `type="log"`.

### `aras/ui/static/index.html`
Vanilla JS UI layout:
- Left: agent status board (IDLE/WORKING/DONE/ERROR) with pulsing indicators
- Center: tabs
  - Logs (color-coded by agent)
  - Paper preview (`paper.tex` excerpt)
  - Memory preview (RAG excerpt + prompt version)
  - GitHub (repo URL link when available)
- Right: progress pipeline with percent bars
- Bottom: task + token/cost/budget counters + paper score

### WebSocket message schema

State snapshots:

```json
{
  "type": "state",
  "state": {
    "topic": "string|null",
    "cycle": 1,
    "agents": { "agent_id": { "status": "IDLE|WORKING|DONE|ERROR", "last_update": "...", "detail": "..." } },
    "pipeline": [ { "name": "Novelty", "progress": 0.0 } ],
    "current_task": "string",
    "tokens_used": 0,
    "tokens_input": 0,
    "tokens_output": 0,
    "errors": 0,
    "cost_usd": 0.0,
    "budget_remaining_usd": null,
    "cost_per_cycle": 0.0,
    "paper_preview": "string",
    "memory_preview": "string",
    "github_url": "string|null",
    "hf_url": "string|null",
    "paper_score": 6.5
  }
}
```

Log events:

```json
{
  "type": "log",
  "event": {
    "agent": "orchestrator|research|scraping|citations|novelty|coder|analyst|figures|coherence|writer|reviewer|memory|ab_tester|hf|github",
    "message": "string",
    "level": "info|error",
    "ts": "iso8601"
  }
}
```

## Logging and artifacts

### Logging
- Console logging via Rich
- File logging to `logs/aras.log`
- Analysis artifact: `logs/analysis.json`
- Novelty evidence artifact: `logs/novelty_evidence_cycle{N}.json`
- Objective cycle ledger: `logs/cycle_quality.jsonl`

### Primary output directories
- `experiments/<topic_slug>/`
  - `manifest.json`
  - per-experiment run folders (each containing `results.json`, `loss.png`, `loss.pdf`)
  - aggregate `results.json`
- `paper/`
  - `paper.tex`
  - `references.bib`
  - `paper.pdf` (if compiled)
- `chroma_db/` (Chroma persistence)
- `prompt_versions/prompts_vN.json`
- `logs/cost_log.jsonl` (LLM cost ledger, one JSONL line per recorded call)
- `memory_snapshot/chroma_db.zip` (created during per-cycle and periodic snapshot loops; publishing is independent)
- `paper/diffs/paper_diff_round{N}.patch` (unified diffs for reviewer/coherence revision rounds)
- `TEST-CONTRACT.md` (functional truth contract used by integration tests)

## Docker / Compose

### `aras/docker-compose.yml`
Services:
- `redis`: cache backend
- `chroma`: ChromaDB server container (persistence mounted to `./chroma_db`)
- `local_model_1/2/3`: llama-cpp-python OpenAI-compatible servers
- `aras`: ARAS container (builds from `aras/Dockerfile`)

Important notes:
- The compose file expects model files to be present in `./models/`:
  - `/models/model1.gguf`, `/models/model2.gguf`, `/models/model3.gguf`
- The container supports additional optional env vars for enhanced functionality:
  - `HF_TOKEN`, `HF_USERNAME` (Hugging Face dataset publishing)
  - `CROSSREF_EMAIL` (Crossref contact for citation validation)
  - `APPROVAL_WEBHOOK_URL` (human approval gate before publishing)
- ARAS container runs:
  - `python -m aras.main --topic "Docker default topic"`

### `aras/Dockerfile`
- Base: `python:3.12-slim`
- Installs TeX toolchain packages for PDF compilation
- Installs Python deps from `aras/requirements.txt`

## Local runbooks

### Windows (recommended)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r .\aras\requirements.txt
Copy-Item .\aras\.env.example .\aras\.env -ErrorAction SilentlyContinue

# Optional: if :8000 is already in use, choose another UI port (example: 8001).
# $env:UI_PORT=8001

py -m aras.main --topic "Your Topic"
```

### PowerShell helper
- `aras/quick_start.ps1` performs the above steps and runs ARAS.

### Linux/macOS helper
- `aras/quick_start.sh` does the same for Unix-like environments.

## Testing (independent components)

The system is structured so each agent can be instantiated and called independently (they accept `Settings`, `MemoryAgent`, and an event sink).

Pytest suite highlights:
- **Offline-by-default smoke tests** for UI endpoints, runner, templates, config, FiguresAgent, HuggingFaceAgent (mocked).
- **Opt-in markers**:
  - `network`: requires network + relevant env vars
  - `llm`: requires LLM keys
  - `slow`: runs slower integration-like checks
- **Mini e2e**: a slow test executes a real experiment as a subprocess and asserts `results.json` + `loss.png`.
- **Functional contract integration tests** now validate that behavior is connected and real (not UI-only):
  - strict novelty gate blocks/permits pivot correctly
  - cycle quality ledger is written and parseable
  - UI quality endpoint reflects on-disk quality ledger
  - figures endpoint reflects real PNG artifacts and structured captions
- **Opt-in novelty network integration test** validates live novelty evidence ingestion from real APIs.

Experiment tests:
- Each generated experiment bundle includes:
  - `experiments/<slug>/tests/test_experiments_import.py`

Run tests (after installing requirements):

```powershell
py -m pytest -q
```

Run only slow tests:

```powershell
py -m pytest tests -m slow -v
```

Run only network tests:

```powershell
py -m pytest tests/network -m network -v
```

Network test notes:
- `tests/network/test_novelty_network.py` is intentionally opt-in.
- Requires internet connectivity.
- Requires `CROSSREF_EMAIL` for Crossref-compliant requests.

Note: the CLI flag `--timeout=...` requires the optional `pytest-timeout` plugin (see `tests/requirements_test.txt`).

## UI approval workflow (local)

ARAS uses a webhook-style approval gate:
- At publish time, `ApprovalGate` sends a POST to `APPROVAL_WEBHOOK_URL` with a generated `request_id`.
- The UI backend stores this as a **pending** approval and blocks until a decision arrives (or timeout).
- The UI polls `GET /api/approval/pending` and only enables approval actions when a real `request_id` exists.

Operational tips:
- If you run the UI on a non-default port (e.g., `UI_PORT=8001`), also set:
  - `APPROVAL_WEBHOOK_URL=http://127.0.0.1:8001/api/approval/webhook`
- If Approve/Reject appears “unresponsive”, check:
  - `GET http://127.0.0.1:<port>/api/approval/pending` → if it returns `{"pending": false}` there is nothing to approve yet.

## Troubleshooting (local dry run)

- **UI won’t start: WinError 10048 / port already in use**
  - Another process is holding the port. Either stop it or start ARAS with `UI_PORT=8001` (or any free port).

- **Local models**
  - You do **not** need a separate `llama.cpp` folder on PATH.
  - ARAS starts local servers via `python -m llama_cpp.server` from the `llama-cpp-python` package.

- **First run seems “stuck” early**
  - Chroma may download an embedding model (`all-MiniLM-L6-v2`) on first boot; that can take time and looks like a slow progress bar in logs.

## Known limitations vs. the original spec

The implementation focuses on a **fully runnable** ARAS with real UI, persistence, and best-effort publishing, but a few original “production-grade autonomy” items remain pragmatic or partial:
- **Domain-appropriate experiments**: experiments are CPU-only and run under strict timeouts, using real publicly available datasets when possible (with offline fallbacks via `sklearn.datasets` and targeted synthetic generation when data sources are unavailable). Experiments stream live metrics and include targeted self-healing (auto-install missing deps and simplify dataset/training parameters on failure).
- **Citation validation scope**: Crossref enrichment is applied, including DOI/title resolution and retraction heuristics, but it does not guarantee full publisher metadata normalization across all edge cases.
- **Hugging Face publishing**: Hugging Face publishing creates a dataset repo, and (when checkpoint artifacts are detected) also creates a model repo. It always attempts to create and upload a Gradio Space demo (with auto-generated `app.py`), polls the Space build stage, and persists the latest published URLs to `logs/hf_last.json`.
- **Cost accuracy**: pricing is best-effort via a static table and token splits; if a provider/model has different pricing than the table, reported USD can deviate.
- **UI depth**: the UI includes dedicated `DIFFS` and `FIGURES` tabs. Diffs are rendered client-side via pure DOM from unified patches, and figures are presented in a gallery with lightbox zoom; the backend runs background watchers that broadcast `diff` and `figure_ready` events over WebSockets.
- **Approval webhook contract**: publishing requires the webhook to return a JSON with an `approved` boolean (and optional `note`); invalid responses fall back to rejection.
