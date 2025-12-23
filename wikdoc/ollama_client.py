"""Minimal Ollama HTTP client helpers.

Wikdoc uses Ollama for:
  - embeddings: POST /api/embeddings
  - chat/generation: POST /api/chat

This module contains lightweight health-check helpers used by the TUI menu.
"""

from __future__ import annotations

import requests


def check_ollama(host: str = "http://localhost:11434", timeout: int = 3) -> bool:
    """Return True if Ollama is reachable.

    We try a couple endpoints because Ollama versions vary a bit.
    """
    base = host.rstrip("/")
    # Common endpoints: /api/tags (list models) and /api/version
    for path in ("/api/tags", "/api/version"):
        try:
            r = requests.get(base + path, timeout=timeout)
            if 200 <= r.status_code < 300:
                return True
        except Exception:
            continue
    return False


def list_models(host: str = "http://localhost:11434", timeout: int = 5) -> list[str]:
    """List installed models (best-effort)."""
    base = host.rstrip("/")
    r = requests.get(base + "/api/tags", timeout=timeout)
    r.raise_for_status()
    data = r.json()
    models = []
    for m in data.get("models", []):
        name = m.get("name")
        if name:
            models.append(name)
    return models
