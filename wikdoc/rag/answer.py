"""LLM answering backends."""

from __future__ import annotations

import requests
import os
from dataclasses import dataclass, field
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
    timeout: int = field(default_factory=lambda: int(os.getenv("WIKDOC_CHAT_TIMEOUT", "300")))

    def chat(self, messages: List[dict]) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        r = requests.post(self.host.rstrip("/") + "/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "").strip()
