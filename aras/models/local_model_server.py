from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from aras.config import Settings
from aras.utils.logging import get_logger


log = get_logger("local-model")


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            return s.connect_ex((host, port)) == 0
        except Exception:
            return False


@dataclass
class LocalServerSpec:
    model_path: str
    port: int


class LocalModelServerManager:
    """Start/stop llama-cpp-python OpenAI-compatible servers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._procs: list[subprocess.Popen[str]] = []

    async def start_all(self) -> None:
        """Start 3 servers if not already up."""
        specs = [
            LocalServerSpec(self.settings.local_model_1_path, self.settings.local_model_ports[0]),
            LocalServerSpec(self.settings.local_model_2_path, self.settings.local_model_ports[1]),
            LocalServerSpec(self.settings.local_model_3_path, self.settings.local_model_ports[2]),
        ]
        for spec in specs:
            if _port_open(self.settings.local_model_host, spec.port):
                continue
            await self._start_one(spec)

    async def _start_one(self, spec: LocalServerSpec) -> None:
        model_path = Path(spec.model_path)
        if not model_path.exists():
            log.warning("Local model file missing: %s", model_path)
            return
        cmd = [
            os.environ.get("PYTHON", sys.executable),
            "-m",
            "llama_cpp.server",
            "--model",
            str(model_path),
            "--host",
            self.settings.local_model_host,
            "--port",
            str(spec.port),
        ]
        log.info("Starting local model server: %s", " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self._procs.append(proc)
        await asyncio.sleep(1.0)

    async def stop_all(self) -> None:
        """Stop all started servers."""
        for p in self._procs:
            try:
                p.terminate()
            except Exception:
                pass


class LocalModelClient:
    """Client for local OpenAI-compatible llama-cpp servers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _pick_base_url(self, model: str) -> str:
        # model can be "local" (default) or "local-coder"/etc; map to ports.
        mapping = {
            "local": self.settings.local_model_ports[0],
            "local-coder": self.settings.local_model_ports[0],
            "local-analyst": self.settings.local_model_ports[1],
            "local-writer": self.settings.local_model_ports[2],
        }
        port = mapping.get(model, self.settings.local_model_ports[0])
        return f"http://{self.settings.local_model_host}:{port}/v1"

    async def chat(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int, int, int]:
        base = self._pick_base_url(model)
        url = f"{base}/chat/completions"
        payload: dict[str, Any] = {
            "model": "gpt-3.5-turbo",  # llama-cpp ignores name but expects one
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        # Best-effort split for llama-cpp server variants that only expose total_tokens.
        if prompt_tokens == 0 and completion_tokens == 0 and total_tokens:
            prompt_tokens = total_tokens
            completion_tokens = 0
        return text, prompt_tokens, completion_tokens, total_tokens

