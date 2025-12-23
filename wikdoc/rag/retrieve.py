"""Retrieval utilities.

This module contains small helpers around the vector store search operation.
"""

from __future__ import annotations

from typing import List, Sequence

from ..vectordb.base import SearchHit, VectorStore


def retrieve(store: VectorStore, query_vec: Sequence[float], top_k: int) -> List[SearchHit]:
    """Retrieve top-k chunks from the vector store."""
    return store.search(query_vec=query_vec, top_k=top_k)
