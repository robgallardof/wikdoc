# wikdoc/ingest/ignore_rules.py
"""Ignore rules for workspace scanning."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class IgnoreMatcher:
    """
    Matches paths that should be excluded from indexing.
    """

    root: Path
    exclude_globs: List[str]
    gitignore_patterns: List[str]

    def matches(self, path: Path) -> bool:
        """
        Return True if the given path should be ignored.

        Args:
            path: File path to check.

        Returns:
            True if excluded by globs/gitignore.
        """
        rel = str(path.relative_to(self.root)).replace("\\", "/")

        for pat in self.exclude_globs:
            if fnmatch.fnmatch(rel, pat):
                return True

        if self.gitignore_patterns:
            try:
                import pathspec  # type: ignore

                spec = pathspec.PathSpec.from_lines("gitwildmatch", self.gitignore_patterns)
                return spec.match_file(rel)
            except Exception:
                # naive fallback
                for pat in self.gitignore_patterns:
                    pat = pat.strip()
                    if not pat or pat.startswith("#"):
                        continue
                    if pat.endswith("/") and rel.startswith(pat):
                        return True
                    if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, f"**/{pat}"):
                        return True

        return False


def load_gitignore(root: Path) -> List[str]:
    """
    Load `.gitignore` patterns from the workspace root if present.

    Args:
        root: Workspace root folder.

    Returns:
        List of patterns (raw lines).
    """
    gi = root / ".gitignore"
    if not gi.exists():
        return []
    try:
        return gi.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []


def build_ignore_matcher(root: Path, exclude_globs: List[str], use_gitignore: bool = True) -> IgnoreMatcher:
    """
    Create an IgnoreMatcher for a workspace.

    Args:
        root: Workspace root.
        exclude_globs: Glob patterns to ignore.
        use_gitignore: Whether to use .gitignore as additional ignore source.

    Returns:
        IgnoreMatcher instance.
    """
    patterns = load_gitignore(root) if use_gitignore else []
    return IgnoreMatcher(root=root, exclude_globs=exclude_globs, gitignore_patterns=patterns)
