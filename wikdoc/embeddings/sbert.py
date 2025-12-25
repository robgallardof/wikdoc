# wikdoc/embeddings/sbert.py
"""SentenceTransformers embedding backend (optional)."""

from __future__ import annotations

from typing import List, Sequence

from .base import Embedder


class SentenceTransformersEmbedder(Embedder):
    """
    Embeddings via `sentence-transformers`.
    """

    def __init__(self, model_name: str = "intfloat/e5-small-v2") -> None:
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace model id.

        Raises:
            RuntimeError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError("sentence-transformers is not installed. Install with `pip install wikdoc[st]`.") from e
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[Sequence[float]]:
        """
        Embed texts into vectors.

        Args:
            texts: Input strings.

        Returns:
            Normalized embeddings aligned to input.
        """
        vecs = self._model.encode(texts, normalize_embeddings=True).tolist()
        return vecs
