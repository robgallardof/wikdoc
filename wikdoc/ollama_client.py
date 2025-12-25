# wikdoc/ollama/client.py
"""Minimal Ollama HTTP client helpers."""

from __future__ import annotations

from typing import List

import requests


def check_ollama(host: str = "http://localhost:11434", timeout: int = 3) -> bool:
    """
    Return True if Ollama is reachable.

    Args:
        host: Base URL.
        timeout: Timeout seconds.

    Returns:
        True if a known endpoint returns 2xx.
    """
    base = host.rstrip("/")
    for path in ("/api/tags", "/api/version"):
        try:
            r = requests.get(base + path, timeout=timeout)
            if 200 <= r.status_code < 300:
                return True
        except Exception:
            continue
    return False


def list_models(host: str = "http://localhost:11434", timeout: int = 5) -> List[str]:
    """
    List installed models (best-effort).

    Args:
        host: Base URL.
        timeout: Timeout seconds.

    Returns:
        List of model names.

    Raises:
        requests.HTTPError: On non-2xx response.
    """
    base = host.rstrip("/")
    r = requests.get(base + "/api/tags", timeout=timeout)
    r.raise_for_status()
    data = r.json() or {}
    models: List[str] = []
    for m in data.get("models", []):
        name = m.get("name")
        if name:
            models.append(name)
    return models
