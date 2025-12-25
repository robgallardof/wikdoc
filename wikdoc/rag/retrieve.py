# wikdoc/rag/retrieve.py
"""Retrieval utilities."""

from __future__ import annotations

from typing import List, Sequence

from ..vectordb.base import SearchHit, VectorStore


def retrieve(store: VectorStore, query_vec: Sequence[float], top_k: int) -> List[SearchHit]:
    """
    Retrieve top-k chunks from the vector store.

    Args:
        store: Vector store.
        query_vec: Query embedding vector.
        top_k: Number of hits.

    Returns:
        List of SearchHit.
    """
    return store.search(query_vec=query_vec, top_k=top_k)
