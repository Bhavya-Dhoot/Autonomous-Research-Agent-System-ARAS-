from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from aras.config import Settings
from aras.models.nvidia_client import NvidiaClient
from aras.models.openai_client import OpenAIClient
from aras.models.local_model_server import LocalModelClient
from aras.utils.logging import get_logger


log = get_logger("fallback")


Provider = Literal["local", "nvidia", "openai"]


@dataclass
class ChatResult:
    text: str
    provider: Provider
    model: str
    tokens_input: int
    tokens_output: int
    tokens_total: int
    # Backward-compat for existing token accounting code.
    tokens_used: int


class FallbackRouter:
    """Route chat completions with fault-tolerant fallback logic."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.local = LocalModelClient(settings=settings)
        self.nvidia = NvidiaClient(settings=settings)
        self.openai = OpenAIClient(settings=settings)

    async def chat(
        self,
        *,
        role_system: str,
        messages: list[dict[str, str]],
        purpose: str,
        prefer: list[Provider],
        model_overrides: dict[Provider, str] | None = None,
        thinking: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ChatResult:
        """Try providers in order with retries and backoff."""
        model_overrides = model_overrides or {}

        last_exc: Exception | None = None
        for provider in prefer:
            try:
                async for attempt in AsyncRetrying(
                    reraise=True,
                    stop=stop_after_attempt(self.settings.max_retries),
                    wait=wait_exponential_jitter(
                        initial=self.settings.backoff_min_seconds, max=self.settings.backoff_max_seconds
                    ),
                    retry=retry_if_exception_type(Exception),
                ):
                    with attempt:
                        if provider == "local":
                            model = model_overrides.get("local", "local")
                            txt, ti, to, tt = await self.local.chat(
                                system=role_system,
                                messages=messages,
                                model=model,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            return ChatResult(
                                text=txt,
                                provider="local",
                                model=model,
                                tokens_input=ti,
                                tokens_output=to,
                                tokens_total=tt,
                                tokens_used=tt,
                            )
                        if provider == "nvidia":
                            model = model_overrides.get("nvidia", self.settings.nvidia_default_model)
                            txt, ti, to, tt = await self.nvidia.chat(
                                model=model,
                                system=role_system,
                                messages=messages,
                                thinking=thinking,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            return ChatResult(
                                text=txt,
                                provider="nvidia",
                                model=model,
                                tokens_input=ti,
                                tokens_output=to,
                                tokens_total=tt,
                                tokens_used=tt,
                            )
                        if provider == "openai":
                            model = model_overrides.get("openai", self.settings.openai_orchestrator_model)
                            txt, ti, to, tt = await self.openai.chat(
                                model=model,
                                system=role_system,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            return ChatResult(
                                text=txt,
                                provider="openai",
                                model=model,
                                tokens_input=ti,
                                tokens_output=to,
                                tokens_total=tt,
                                tokens_used=tt,
                            )
                        raise RuntimeError(f"unknown provider: {provider}")
            except Exception as e:
                last_exc = e
                log.warning("Provider failed (%s) purpose=%s err=%s", provider, purpose, e)
                await asyncio.sleep(0.2)
                continue
        raise RuntimeError(f"All providers failed for purpose={purpose}: {last_exc}")

