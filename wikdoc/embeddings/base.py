"""Embedding interfaces."""

from __future__ import annotations

from typing import List, Sequence


class Embedder:
    """Embedder interface for turning text into vectors."""

    def embed(self, texts: List[str]) -> List[Sequence[float]]:
        """Return embeddings for each input text."""
        raise NotImplementedError
