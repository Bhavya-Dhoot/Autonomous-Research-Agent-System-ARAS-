from __future__ import annotations

import os

import pytest

from aras.healing.fallback_router import FallbackRouter


@pytest.mark.llm
@pytest.mark.asyncio
async def test_llm_opt_in_smoke(settings) -> None:
    """
    Opt-in smoke: if you set NVIDIA_API_KEY or OPENAI_API_KEY, ensure the router can complete one call.
    This test is skipped unless keys are set.
    """
    if not (os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        pytest.skip("No LLM API key set for opt-in test")

    router = FallbackRouter(settings=settings)
    res = await router.chat(
        role_system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Say OK"}],
        purpose="smoke",
        prefer=["nvidia", "openai", "local"],
        thinking=False,
        temperature=0.0,
        max_tokens=16,
    )
    assert isinstance(res.text, str) and len(res.text) > 0

