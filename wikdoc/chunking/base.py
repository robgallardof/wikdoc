"""Chunking interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    """A chunk of text extracted from a document.

    Attributes:
        text: Chunk content.
        start_line: 1-based start line in the source file (best effort).
        end_line: 1-based end line in the source file (best effort).
        symbol: Optional symbol name (class/method/function) if structure-aware.
    """

    text: str
    start_line: int
    end_line: int
    symbol: Optional[str] = None


class Chunker:
    """Chunker interface."""

    def chunk(self, text: str) -> List[Chunk]:
        """Split document text into chunks."""
        raise NotImplementedError
