"""Document loading utilities.

Wikdoc indexes *text* files only. This module includes:
  - best-effort text/binary sniffing
  - safe reads with encoding fallback

It intentionally avoids heavy dependencies to keep Wikdoc portable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


def is_probably_binary(data: bytes) -> bool:
    """Heuristic binary detection."""
    if not data:
        return False
    if b"\x00" in data:
        return True
    text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
    nontext = data.translate(None, text_chars)
    return float(len(nontext)) / float(len(data)) > 0.30


def read_text_file(path: Path, max_bytes: int) -> Tuple[str, str]:
    """Read file content up to `max_bytes`.

    Args:
        path: File path.
        max_bytes: Maximum bytes to read.

    Returns:
        Tuple of (content, encoding_used).

    Raises:
        ValueError: If file appears to be binary.
    """
    raw = path.read_bytes()[:max_bytes]
    if is_probably_binary(raw):
        raise ValueError("Binary file detected")
    try:
        return raw.decode("utf-8"), "utf-8"
    except Exception:
        return raw.decode("latin-1"), "latin-1"
