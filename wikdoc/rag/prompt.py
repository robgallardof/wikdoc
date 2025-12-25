# wikdoc/rag/prompt.py
"""Prompt building for RAG Q&A."""

from __future__ import annotations

from typing import List, Dict

from ..vectordb.base import SearchHit


SYSTEM_PROMPT = (
    "You are Wikdoc, a local documentation assistant.\n"
    "Use ONLY the provided context.\n"
    "If the answer is not supported by the context, say you don't know and suggest where to look.\n"
    "Always include citations in the format [path:start-end] for claims derived from context.\n"
    "Be concise and technical.\n"
)


def format_context(hits: List[SearchHit], max_chars: int) -> str:
    """
    Format retrieved hits into a context block with file/line headers.

    Args:
        hits: Retrieved results.
        max_chars: Context cap in characters.

    Returns:
        Context string.
    """
    parts: List[str] = []
    total = 0
    for h in hits:
        header = f"FILE: {h.path} (lines {h.start_line}-{h.end_line})\n"
        body = (h.text or "").strip()
        block = header + body + "\n\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts).strip()


def build_prompt(question: str, hits: List[SearchHit], max_context_chars: int) -> List[Dict[str, str]]:
    """
    Build an Ollama-style chat payload.

    Args:
        question: User question.
        hits: Retrieved context hits.
        max_context_chars: Context cap.

    Returns:
        List of chat messages.
    """
    ctx = format_context(hits, max_context_chars)
    user = (
        "CONTEXT:\n" + ctx + "\n\n"
        "QUESTION:\n" + question + "\n\n"
        "Answer with citations like [path:start-end]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
