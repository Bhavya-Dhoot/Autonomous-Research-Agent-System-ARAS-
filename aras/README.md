# ARAS — Autonomous Research Agent System

ARAS is an end-to-end autonomous research pipeline that (best-effort) plans, scrapes sources, runs experiments, writes an IEEE/ACM-style LaTeX paper, reviews it, persists memory (ChromaDB), self-improves prompts, and optionally publishes outputs to GitHub — while streaming real-time state in a FastAPI WebSocket UI.

## Quick start (Windows)

1. Create `.env` from `.env.example` and set keys (optional but recommended):
   - `OPENAI_API_KEY`, `NVIDIA_API_KEY`, `GITHUB_TOKEN`
   - Optional publishing: `HF_TOKEN`, `HF_USERNAME`
   - Optional citations: `CROSSREF_EMAIL`
   - Human approval gate: `APPROVAL_WEBHOOK_URL` (if set, publishing waits for approval)
2. Install Python deps and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r .\aras\requirements.txt
py .\aras\main.py --topic "Your Research Topic"
```

Then open `http://localhost:8000`.

## Docker Compose

This repo includes a `docker-compose.yml` that brings up:
- Redis
- ChromaDB server
- 3 local OpenAI-compatible model servers (llama-cpp-python)
- FastAPI UI + orchestrator

You must mount model `.gguf` files where specified.

```bash
docker-compose up --build
```

## Notes

- If no model keys are configured, ARAS will still run using deterministic fallbacks (template-based writing and lightweight experiments), and will log missing capabilities to memory for later improvement.
- Paper compilation uses `tectonic` when available, otherwise `pdflatex`. In Docker, a TeX engine is installed.

