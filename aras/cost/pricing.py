from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class PricingTier:
    """Token pricing for a specific provider/model pairing."""

    input_per_million_usd: float
    output_per_million_usd: float


# Best-effort pricing table for ARAS v2 cost accounting.
# Values are per 1M tokens (USD).
#
# Notes:
# - Local models are treated as $0.00 (no API token cost).
# - If a model is missing, cost will default to $0.00 for that provider/model.
PRICING_TABLE: Final[dict[str, PricingTier]] = {
    # OpenAI
    # GPT-4.1: input $2.00 / 1M, output $8.00 / 1M
    "openai:gpt-4.1": PricingTier(input_per_million_usd=2.00, output_per_million_usd=8.00),
    # GPT-4o: input $2.50 / 1M, output $10.00 / 1M (best-effort)
    "openai:gpt-4o": PricingTier(input_per_million_usd=2.50, output_per_million_usd=10.00),
    # NVIDIA (moonshotai/kimi-k2.5) best-effort pricing.
    "nvidia:moonshotai/kimi-k2.5": PricingTier(input_per_million_usd=0.60, output_per_million_usd=2.00),
}


def price_key(provider: str, model: str) -> str:
    """Create a stable lookup key."""
    return f"{provider}:{model}"


def get_pricing(provider: str, model: str) -> PricingTier:
    """Return pricing for provider/model or $0.00 default."""
    key = price_key(provider, model)
    return PRICING_TABLE.get(key, PricingTier(input_per_million_usd=0.0, output_per_million_usd=0.0))

