from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    # Load `.env` from either repo root or `aras/.env` (when present).
    model_config = SettingsConfigDict(
        # Prefer `aras/.env` when present, otherwise fall back to repo-root `.env`.
        env_file=(Path(__file__).resolve().parent / ".env", Path(".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys / tokens
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    nvidia_api_key: str | None = Field(default=None, alias="NVIDIA_API_KEY")
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    github_owner: str | None = Field(default=None, alias="GITHUB_OWNER")
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    hf_username: str | None = Field(default=None, alias="HF_USERNAME")
    crossref_email: str | None = Field(default=None, alias="CROSSREF_EMAIL")

    # Budget / cost
    budget_usd: float | None = Field(default=None, alias="BUDGET_USD")
    budget_usd_ceiling: float = Field(default=5.0, alias="BUDGET_USD_CEILING", ge=0.0)

    # Approval gate
    approval_webhook_url: str | None = Field(default=None, alias="APPROVAL_WEBHOOK_URL")
    approval_timeout_seconds: int = Field(default=3600, alias="APPROVAL_TIMEOUT_SECONDS")

    # Novelty / review / experiments tuning
    review_rounds: int = Field(default=3, alias="REVIEW_ROUNDS")
    max_experiment_timeout_seconds: int = Field(default=300, alias="MAX_EXPERIMENT_TIMEOUT_SECONDS")
    figure_quality_rerun_enabled: bool = Field(default=True, alias="FIGURE_QUALITY_RERUN_ENABLED")
    figure_quality_max_reruns: int = Field(default=2, alias="FIGURE_QUALITY_MAX_RERUNS", ge=0)
    novelty_pivot_max_score: float = Field(default=0.35, alias="NOVELTY_PIVOT_MAX_SCORE", ge=0.0, le=1.0)
    novelty_min_confidence: float = Field(default=0.65, alias="NOVELTY_MIN_CONFIDENCE", ge=0.0, le=1.0)
    novelty_min_validated_evidence: int = Field(default=3, alias="NOVELTY_MIN_VALIDATED_EVIDENCE", ge=1)
    novelty_min_evidence_sources: int = Field(default=2, alias="NOVELTY_MIN_EVIDENCE_SOURCES", ge=1)

    # Local model servers (llama-cpp)
    local_model_1_path: str = Field(default="./models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf", alias="LOCAL_MODEL_1_PATH")
    local_model_2_path: str = Field(default="./models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", alias="LOCAL_MODEL_2_PATH")
    local_model_3_path: str = Field(default="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", alias="LOCAL_MODEL_3_PATH")
    local_model_host: str = Field(default="127.0.0.1", alias="LOCAL_MODEL_HOST")
    local_model_port_base: int = Field(default=8100, alias="LOCAL_MODEL_PORT_BASE")

    # Persistence / infra
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")

    # UI / server
    ui_host: str = Field(default="127.0.0.1", alias="UI_HOST")
    ui_port: int = Field(default=8000, alias="UI_PORT")
    ws_path: str = "/ws/status"
    auto_open_browser: bool = Field(default=True, alias="AUTO_OPEN_BROWSER")

    # Models / routing
    nvidia_default_model: str = "moonshotai/kimi-k2.5"
    openai_orchestrator_model: str = "gpt-4.1"  # best-effort default; change if needed
    openai_figures_model: str = "gpt-4o"

    local_model_ids: tuple[str, str, str] = ("local-coder", "local-analyst", "local-writer")
    local_model_ports: tuple[int, int, int] = (0, 0, 0)

    # Scraping / cache
    scrape_cache_ttl_seconds: int = Field(default=24 * 60 * 60, alias="SCRAPE_CACHE_TTL_SECONDS")

    # Paths
    data_dir: str = "./data"
    logs_dir: str = "./logs"
    memory_snapshot_dir: str = "./memory_snapshot"
    memory_snapshot_interval_seconds: int = Field(default=900, alias="MEMORY_SNAPSHOT_INTERVAL_SECONDS")

    paper_dir: str = "./paper"
    experiments_dir: str = "./experiments"

    # Health / retries
    heartbeat_interval_seconds: int = Field(default=30, alias="HEARTBEAT_INTERVAL_SECONDS")
    health_restart_grace_seconds: int = 90

    max_retries: int = 6
    backoff_min_seconds: float = 1.0
    backoff_max_seconds: float = 60.0

    routing_escalation_failures: int = 2
    openai_escalation_model: str = "gpt-4o"

    # Prompt / run-mode
    prompt_store_path: str = "./prompt_versions"
    run_mode: Literal["local", "docker"] = "local"

    def resolved_paths(self) -> dict[str, Path]:
        """Resolve key filesystem paths relative to current working directory."""
        return {
            "chroma_persist_dir": Path(self.chroma_persist_dir).resolve(),
            "data_dir": Path(self.data_dir).resolve(),
            "logs_dir": Path(self.logs_dir).resolve(),
            "memory_snapshot_dir": Path(self.memory_snapshot_dir).resolve(),
            "paper_dir": Path(self.paper_dir).resolve(),
            "experiments_dir": Path(self.experiments_dir).resolve(),
            "prompt_store_path": Path(self.prompt_store_path).resolve(),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    s = Settings()
    base = s.local_model_port_base
    s.local_model_ports = (base, base + 1, base + 2)
    return s
