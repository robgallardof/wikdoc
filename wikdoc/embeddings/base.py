# wikdoc/embeddings/base.py
"""Embedding interfaces."""

from __future__ import annotations

from typing import List, Sequence


class Embedder:
    """
    Embedder interface for turning text into vectors.
    """

    def embed(self, texts: List[str]) -> List[Sequence[float]]:
        """
        Return embeddings for each input text.

        Args:
            texts: List of input strings.

        Returns:
            List of vectors aligned to `texts`.

        Raises:
            NotImplementedError: If not implemented.
        """
        raise NotImplementedError
