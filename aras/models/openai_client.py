from __future__ import annotations

from typing import Any

import httpx

from aras.config import Settings
from aras.utils.logging import get_logger


log = get_logger("openai")


class OpenAIClient:
    """Minimal async OpenAI Chat Completions client wrapper."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def chat(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int, int, int]:
        """Call OpenAI API (best-effort)."""
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        # Some providers only return total_tokens; keep best-effort split.
        if prompt_tokens == 0 and completion_tokens == 0 and total_tokens:
            prompt_tokens = total_tokens
            completion_tokens = 0
        return text, prompt_tokens, completion_tokens, total_tokens

