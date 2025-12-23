"""Ollama embedding backend.

This embedder calls the local Ollama HTTP API to produce embeddings.
It is intentionally defensive:
  - sanitizes control characters
  - truncates very long inputs (configurable)
  - retries with smaller inputs if Ollama returns 500
"""

from __future__ import annotations

import os
import time
from typing import List

import requests


def _sanitize_text(s: str) -> str:
    """Sanitize text to reduce embedding backend crashes."""
    if not isinstance(s, str):
        s = str(s)

    # Remove NULs (can crash tokenizers)
    s = s.replace("\x00", "")

    # Normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    return s


class OllamaEmbedder:
    """Compute embeddings via Ollama's HTTP API."""

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model
        self.max_chars = int(os.getenv("WIKDOC_EMBED_MAX_CHARS", "4000"))
        self.min_chars = int(os.getenv("WIKDOC_EMBED_MIN_CHARS", "800"))
        self.timeout = int(os.getenv("WIKDOC_EMBED_TIMEOUT", "180"))

    def _post_embeddings(self, text: str) -> requests.Response:
        payload = {"model": self.model, "input": text}
        r = requests.post(f"{self.host}/api/embeddings", json=payload, timeout=self.timeout)
        if r.status_code == 404:
            payload = {"model": self.model, "prompt": text}
            r = requests.post(f"{self.host}/api/embeddings", json=payload, timeout=self.timeout)
        return r

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for raw in texts:
            t = _sanitize_text(raw)
            if self.max_chars > 0 and len(t) > self.max_chars:
                t = t[: self.max_chars]

            attempt = 0
            cur = t

            while True:
                attempt += 1
                r = self._post_embeddings(cur)

                if 200 <= r.status_code < 300:
                    data = r.json()
                    out.append(data["embedding"])
                    break

                if r.status_code >= 500 and len(cur) > self.min_chars and attempt <= 4:
                    time.sleep(0.5 * attempt)
                    cur = cur[: max(self.min_chars, len(cur) // 2)]
                    continue

                detail = ""
                try:
                    detail = r.text
                except Exception:
                    detail = "<no body>"

                raise requests.HTTPError(
                    f"Ollama embeddings failed (status={r.status_code}).\n"
                    f"Model: {self.model}\n"
                    f"Host: {self.host}\n"
                    f"Tip: exclude lockfiles/minified/big generated files, or lower WIKDOC_EMBED_MAX_CHARS.\n"
                    f"Response: {detail[:800]}",
                    response=r,
                )
        return out
