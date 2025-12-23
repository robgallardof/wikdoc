"""LLM answering backends."""

from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import List


class LLMClient:
    """LLM interface."""

    def chat(self, messages: List[dict]) -> str:
        """Execute a chat completion and return assistant text."""
        raise NotImplementedError


@dataclass
class OllamaLLM(LLMClient):
    """LLM backend using Ollama's `/api/chat`."""

    host: str
    model: str

    def chat(self, messages: List[dict]) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        r = requests.post(self.host.rstrip("/") + "/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "").strip()
