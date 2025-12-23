"""Prompt building for RAG Q&A."""

from __future__ import annotations

from typing import List

from ..vectordb.base import SearchHit


SYSTEM_PROMPT = (
    "You are Wikdoc, a local documentation assistant.\n"
    "Use ONLY the provided context.\n"
    "If the answer is not supported by the context, say you don't know and suggest where to look.\n"
    "Always include citations in the format [path:start-end] for claims derived from context.\n"
    "Be concise and technical.\n"
)


def format_context(hits: List[SearchHit], max_chars: int) -> str:
    """Format retrieved hits into a context block with file/line headers."""
    parts: List[str] = []
    total = 0
    for h in hits:
        header = f"FILE: {h.path} (lines {h.start_line}-{h.end_line})\n"
        body = h.text.strip()
        block = header + body + "\n\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


def build_prompt(question: str, hits: List[SearchHit], max_context_chars: int) -> List[dict]:
    """Build an Ollama-style chat payload."""
    ctx = format_context(hits, max_context_chars)
    user = (
        "CONTEXT:\n" + ctx + "\n"
        "QUESTION:\n" + question + "\n\n"
        "Answer with citations like [path:start-end]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
