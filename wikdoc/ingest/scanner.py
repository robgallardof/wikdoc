"""Workspace scanner.

This module provides:
  - candidate file listing (fast pass) for progress bars
  - document loading with hashing for incremental indexing

Notes:
  - Uses IgnoreMatcher (default excludes + optional .gitignore)
  - Filters by extension
  - Skips large files
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List

from ..config import IndexOptions
from .ignore_rules import build_ignore_matcher
from .loaders import read_text_file


@dataclass
class Document:
    """A source document loaded from disk."""

    path: Path
    rel_path: str
    content: str
    encoding: str
    sha256: str
    language: str


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _detect_language_from_suffix(path: Path) -> str:
    suf = path.suffix.lower().lstrip(".")
    if suf == "":
        name = path.name.lower()
        if name == "dockerfile":
            return "dockerfile"
        if name == "makefile":
            return "make"
        return "text"
    return suf


def list_candidate_files(root: Path, opts: IndexOptions) -> List[Path]:
    """List indexable candidate files quickly for progress reporting.

    Args:
        root: Workspace root folder.
        opts: Index options.

    Returns:
        List of file paths that *might* be indexed.
    """
    matcher = build_ignore_matcher(root, opts.exclude_globs, use_gitignore=opts.use_gitignore)
    include_set = {e.lower().lstrip(".") for e in opts.include_ext}
    max_bytes = int(opts.max_file_mb * 1024 * 1024)

    files: List[Path] = []
    for p in root.rglob("*"):
        try:
            if p.is_dir():
                continue
            if (not opts.follow_symlinks) and p.is_symlink():
                continue
            if matcher.matches(p):
                continue

            lang = _detect_language_from_suffix(p)
            if lang not in include_set and lang not in ("dockerfile", "make"):
                continue

            if p.stat().st_size > max_bytes:
                continue

            files.append(p)
        except Exception:
            continue
    return files


def scan_files(root: Path, files: List[Path], opts: IndexOptions) -> Generator[Document, None, None]:
    """Load a specific list of files and yield Document objects."""
    max_bytes = int(opts.max_file_mb * 1024 * 1024)

    for p in files:
        try:
            lang = _detect_language_from_suffix(p)
            content, enc = read_text_file(p, max_bytes)
            rel = str(p.relative_to(root)).replace("\\", "/")
            h = _sha256_text(content)
            yield Document(path=p, rel_path=rel, content=content, encoding=enc, sha256=h, language=lang)
        except Exception:
            continue
