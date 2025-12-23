"""Vector store interfaces.

A vector store is responsible for:
  - Storing chunks (text + metadata) and their embeddings
  - Searching for the most similar chunks for a query embedding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class SearchHit:
    """A retrieved chunk with similarity score."""

    chunk_id: str
    score: float
    path: str
    start_line: int
    end_line: int
    text: str
    language: str
    symbol: Optional[str]


class VectorStore:
    """Vector store interface."""

    def upsert_chunks(self, chunks: List[dict]) -> int:
        """Insert or update chunks.

        Args:
            chunks: List of chunk dicts. Each dict must include:
                - file_path, start_line, end_line, language, symbol (optional)
                - file_sha256, text, embedding

        Returns:
            Number of chunks written.
        """
        raise NotImplementedError

    def search(self, query_vec: Sequence[float], top_k: int) -> List[SearchHit]:
        """Search similar chunks for a query embedding."""
        raise NotImplementedError

    def stats(self) -> dict:
        """Return basic stats about the store."""
        raise NotImplementedError

    def reset(self) -> None:
        """Delete all stored data for this workspace."""
        raise NotImplementedError
