"""Fallback chunker (language-agnostic).

Splits text by character windows with overlap and tries to infer line ranges.

For higher-quality chunking (class/method/function boundaries),
add a Tree-sitter chunker in a future version.
"""

from __future__ import annotations

from typing import List

from .base import Chunk, Chunker


class FallbackChunker(Chunker):
    """Chunker that slices text by character windows."""

    def __init__(self, chunk_chars: int = 6000, overlap_chars: int = 800) -> None:
        self.chunk_chars = max(1000, int(chunk_chars))
        self.overlap_chars = max(0, int(overlap_chars))

    def chunk(self, text: str) -> List[Chunk]:
        lines = text.splitlines(keepends=True)
        offsets = []
        pos = 0
        for i, ln in enumerate(lines, start=1):
            offsets.append((pos, i))
            pos += len(ln)

        def line_at(char_pos: int) -> int:
            last = 1
            for off, li in offsets:
                if off <= char_pos:
                    last = li
                else:
                    break
            return last

        chunks: List[Chunk] = []
        n = len(text)
        start = 0
        while start < n:
            end = min(n, start + self.chunk_chars)
            ctext = text[start:end].strip("\n")
            if ctext:
                sl = line_at(start)
                el = line_at(end)
                chunks.append(Chunk(text=ctext, start_line=sl, end_line=el, symbol=None))
            if end >= n:
                break
            start = max(0, end - self.overlap_chars)
        return chunks
