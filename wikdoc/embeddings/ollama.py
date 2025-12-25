# wikdoc/embeddings/ollama.py
"""
Ollama embedding backend.

This embedder calls the local Ollama HTTP API to produce embeddings.

It is intentionally defensive:
  - sanitizes control characters
  - truncates very long inputs (configurable)
  - retries with smaller inputs if Ollama returns 5xx
  - supports both new and legacy Ollama embedding endpoints

Env vars:
  - WIKDOC_EMBED_MAX_CHARS (default 4000)
  - WIKDOC_EMBED_MIN_CHARS (default 800)
  - WIKDOC_EMBED_TIMEOUT   (default 180)
"""

from __future__ import annotations

import os
import time
from typing import List, Sequence, Optional, Any

import requests


def _sanitize_text(s: str) -> str:
    """
    Sanitize text to reduce embedding backend crashes.

    Args:
        s: Any object convertible to str.

    Returns:
        Cleaned string.
    """
    if not isinstance(s, str):
        s = str(s)

    # Remove NULs (can crash tokenizers)
    s = s.replace("\x00", "")

    # Normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    return s


class OllamaEmbedder:
    """
    Compute embeddings via Ollama's HTTP API.

    Newer Ollama endpoint:
      - POST /api/embed with {"model": "...", "input": ["...","..."]}
      - Response: {"embeddings": [[...], ...]}

    Legacy endpoint:
      - POST /api/embeddings with {"model": "...", "input": "..."} (or "prompt")
      - Response: {"embedding": [...]}

    Attributes:
        host: Ollama base URL.
        model: Embedding model name.
        max_chars: Max characters per input (truncate).
        min_chars: Minimum characters when shrinking on retries.
        timeout: HTTP timeout seconds.
    """

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model
        self.max_chars = int(os.getenv("WIKDOC_EMBED_MAX_CHARS", "4000"))
        self.min_chars = int(os.getenv("WIKDOC_EMBED_MIN_CHARS", "800"))
        self.timeout = int(os.getenv("WIKDOC_EMBED_TIMEOUT", "180"))

    def _post_embed(self, inputs: List[str]) -> requests.Response:
        """
        Call /api/embed (preferred).

        Args:
            inputs: List of texts.

        Returns:
            Response.
        """
        payload = {"model": self.model, "input": inputs}
        return requests.post(f"{self.host}/api/embed", json=payload, timeout=self.timeout)

    def _post_embeddings_legacy(self, text: str) -> requests.Response:
        """
        Call /api/embeddings (legacy).

        Args:
            text: Single text.

        Returns:
            Response.
        """
        payload = {"model": self.model, "input": text}
        r = requests.post(f"{self.host}/api/embeddings", json=payload, timeout=self.timeout)
        if r.status_code == 404:
            payload = {"model": self.model, "prompt": text}
            r = requests.post(f"{self.host}/api/embeddings", json=payload, timeout=self.timeout)
        return r

    @staticmethod
    def _extract_embeddings(data: Any) -> Optional[List[List[float]]]:
        """
        Extract embeddings from Ollama response JSON.

        Args:
            data: Parsed JSON.

        Returns:
            List of embeddings or None.
        """
        if not isinstance(data, dict):
            return None

        embs = data.get("embeddings")
        if isinstance(embs, list) and embs and all(isinstance(x, list) for x in embs):
            return embs  # type: ignore[return-value]

        one = data.get("embedding")
        if isinstance(one, list) and one:
            return [one]  # type: ignore[return-value]

        return None

    def embed(self, texts: List[str]) -> List[Sequence[float]]:
        """
        Generate embeddings for each input string.

        Strategy:
          1) Try /api/embed as a batch.
          2) If unavailable or malformed, fallback to /api/embeddings per item.
          3) Shrink inputs on 5xx errors.

        Args:
            texts: Input strings.

        Returns:
            List of embeddings aligned to texts.

        Raises:
            requests.HTTPError: If Ollama returns persistent non-2xx errors.
            ValueError: If embeddings are empty/malformed.
        """
        if not texts:
            return []

        sanitized: List[str] = []
        for raw in texts:
            t = _sanitize_text(raw)
            if self.max_chars > 0 and len(t) > self.max_chars:
                t = t[: self.max_chars]
            sanitized.append(t)

        # Preferred batch call
        try:
            r = self._post_embed(sanitized)
            if r.status_code != 404:
                r.raise_for_status()
                data = r.json()
                embs = self._extract_embeddings(data)
                if embs and len(embs) == len(sanitized) and all(len(v) > 0 for v in embs):
                    return embs
                raise ValueError("Ollama /api/embed returned empty/malformed embeddings.")
        except Exception:
            # Fall back to legacy per-item
            pass

        out: List[List[float]] = []
        for raw in sanitized:
            attempt = 0
            cur = raw
            while True:
                attempt += 1
                r = self._post_embeddings_legacy(cur)

                if 200 <= r.status_code < 300:
                    data = r.json()
                    embs = self._extract_embeddings(data)
                    if not embs or not embs[0]:
                        raise ValueError("Ollama returned an empty embedding vector.")
                    out.append(embs[0])
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
