"""
LLM answering backends.

This module defines a small abstraction to talk to a "chat LLM" and return the
assistant text only.

Notes:
  - Wikdoc intentionally keeps this interface minimal.
  - Backends should raise on non-2xx responses and return clean text output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict

import requests


class LLMClient:
    """
    Minimal LLM interface used by Wikdoc.

    Implementations must accept an OpenAI/Ollama-style messages array:
      [{"role": "system"|"user"|"assistant", "content": "..."}]

    and return the assistant content as a string.
    """

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Execute a chat completion and return the assistant text.

        Args:
            messages: Chat messages in role/content format.

        Returns:
            Assistant response text (trimmed).

        Raises:
            NotImplementedError: If not implemented by backend.
        """
        raise NotImplementedError


@dataclass
class OllamaLLM(LLMClient):
    """
    LLM backend using Ollama's chat API.

    By default, Ollama uses:
      POST /api/chat

    Environment variables:
      - WIKDOC_CHAT_TIMEOUT: request timeout in seconds (default 600)

    Attributes:
        host: Base URL for Ollama (e.g. http://localhost:11434).
        model: Ollama model name/tag (e.g. qwen2.5-coder:7b).
        timeout: Request timeout in seconds.
    """

    host: str
    model: str
    timeout: int = field(default_factory=lambda: int(os.getenv("WIKDOC_CHAT_TIMEOUT", "600000")))

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Call Ollama and return assistant response content.

        Args:
            messages: Chat messages.

        Returns:
            Assistant text.

        Raises:
            requests.HTTPError: On non-2xx HTTP responses.
            requests.RequestException: On connection/timeout errors.
            ValueError: If response JSON does not contain message content.
        """
        url = f"{self.host.rstrip('/')}/api/chat"
        payload = {"model": self.model, "messages": messages, "stream": False}

        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()

        data = r.json() or {}
        # Ollama chat returns: {"message": {"role":"assistant","content":"..."} , ...}
        content = (data.get("message") or {}).get("content")

        if not isinstance(content, str):
            raise ValueError("Ollama chat response missing 'message.content'.")

        return content.strip()
