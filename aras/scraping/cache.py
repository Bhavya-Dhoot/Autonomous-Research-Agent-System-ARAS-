from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import redis

from aras.config import Settings
from aras.utils.logging import get_logger


log = get_logger("scrape-cache")


@dataclass
class CacheItem:
    key: str
    value: str
    ts: float


class RedisCache:
    """Simple Redis cache wrapper with TTL."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: redis.Redis | None = None

    def _get(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis.from_url(self.settings.redis_url, decode_responses=True)
        return self._client

    @staticmethod
    def make_key(url: str) -> str:
        return "scrape:" + hashlib.sha256(url.encode("utf-8")).hexdigest()

    def get(self, url: str) -> str | None:
        try:
            return self._get().get(self.make_key(url))
        except Exception as e:
            log.warning("Redis get failed: %s", e)
            return None

    def set(self, url: str, value: str) -> None:
        try:
            self._get().set(self.make_key(url), value, ex=self.settings.scrape_cache_ttl_seconds)
        except Exception as e:
            log.warning("Redis set failed: %s", e)

