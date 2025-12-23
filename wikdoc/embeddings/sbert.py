"""SentenceTransformers embedding backend (optional).

Install:
    pip install wikdoc[st]

Notes:
  - Model download happens the first time you run it unless you have cached weights.
  - For a fully offline experience, pre-download the model on your machine.
"""

from __future__ import annotations

from typing import List, Sequence

from .base import Embedder


class SentenceTransformersEmbedder(Embedder):
    """Embeddings via `sentence-transformers`."""

    def __init__(self, model_name: str = "intfloat/e5-small-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "sentence-transformers is not installed. Install with `pip install wikdoc[st]`."
            ) from e
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[Sequence[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True).tolist()
        return vecs
