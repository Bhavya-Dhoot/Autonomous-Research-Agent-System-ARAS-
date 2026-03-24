from __future__ import annotations

import asyncio
import json
from typing import Any

import requests

from aras.config import Settings
from aras.utils.logging import get_logger


log = get_logger("nvidia")


class NvidiaClient:
    """NVIDIA integrate API client with SSE streaming parsing."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def chat(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, str]],
        thinking: bool,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int, int, int]:
        """Call NVIDIA API using the required pattern; executed in a worker thread."""
        if not self.settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY not set")
        return await asyncio.to_thread(
            self._call_nvidia_sync,
            model,
            [{"role": "system", "content": system}, *messages],
            thinking,
            temperature,
            max_tokens,
        )

    def _call_nvidia_sync(
        self,
        model: str,
        messages: list[dict[str, str]],
        thinking: bool = True,
        temperature: float = 1.0,
        max_tokens: int = 16384,
    ) -> tuple[str, int, int, int]:
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
            "Accept": "text/event-stream",
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": 1.0,
            "stream": True,
            "chat_template_kwargs": {"thinking": bool(thinking)},
        }
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        assembled: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue

            choices = obj.get("choices") or []
            usage = obj.get("usage")

            # Some control chunks contain only usage with an empty choices list.
            if not choices:
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens) or total_tokens)
                    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens:
                        prompt_tokens = total_tokens
                        completion_tokens = 0
                continue

            delta = choices[0].get("delta", {}) or {}
            chunk = delta.get("content")
            if chunk:
                assembled.append(chunk)

            if usage:
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens) or total_tokens)
                # If only total_tokens is present, treat it as input.
                if prompt_tokens == 0 and completion_tokens == 0 and total_tokens:
                    prompt_tokens = total_tokens
                    completion_tokens = 0
        return "".join(assembled).strip(), prompt_tokens, completion_tokens, total_tokens

