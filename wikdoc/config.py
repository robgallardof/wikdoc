"""Configuration models and path helpers.

This module centralizes:
  - Workspace identification
  - Storage layout
  - Default ignore patterns / file extensions
  - Index/runtime/backend options

Terminology:
  - Workspace: a folder you want Wikdoc to index.
  - Index: local data produced by Wikdoc (embeddings + metadata).
  - Chunk: a fragment of a file used for retrieval.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


DEFAULT_INCLUDE_EXT: List[str] = [
    # Code
    "py", "js", "ts", "jsx", "tsx", "java", "kt", "cs", "cpp", "c", "h", "hpp",
    "go", "rs", "php", "rb", "swift", "scala", "m", "mm",
    # Project/config/docs
    "json", "yaml", "yml", "toml", "ini", "xml", "properties", "gradle", "csproj",
    "sln", "props", "targets", "config", "env", "md", "txt",
    # Shell/devops
    "sh", "bash", "ps1", "dockerfile", "makefile",
]

DEFAULT_EXCLUDE_GLOBS: List[str] = [
    # Common build/artifact folders
    "**/bin/**", "**/obj/**", "**/.vs/**", "**/.idea/**", "**/.vscode/**",
    "**/node_modules/**", "**/dist/**", "**/build/**", "**/out/**",
    "**/.venv/**",
# Dependency lockfiles / huge generated artifacts (often not useful for Q&A)
"**/pnpm-lock.yaml", "**/package-lock.json", "**/yarn.lock",
"**/*lock*.yaml", "**/*lock*.yml", "**/*lock*.json",
"**/*.min.js", "**/*.min.css", "**/*.map",
 "**/venv/**", "**/__pycache__/**", "**/.pytest_cache/**",
    "**/.git/**", "**/.svn/**", "**/.hg/**",
    # Large/minified assets
    "**/*.min.js", "**/*.min.css",
]


@dataclass(frozen=True)
class Workspace:
    """Represents a folder to index.

    Attributes:
        root: Absolute, normalized path to the workspace root.
        name: Optional friendly name for display/logging.
    """

    root: Path
    name: Optional[str] = None

    @staticmethod
    def from_path(path: str, name: Optional[str] = None) -> "Workspace":
        """Create a workspace from a user-provided path.

        Args:
            path: Folder path to index.
            name: Optional friendly display name.

        Returns:
            Workspace instance.
        """
        p = Path(path).expanduser().resolve()
        return Workspace(root=p, name=name)

    @property
    def id(self) -> str:
        """Stable workspace id derived from absolute path."""
        h = hashlib.sha1(str(self.root).encode("utf-8")).hexdigest()
        return h[:16]


@dataclass
class StoreLayout:
    """Defines where Wikdoc stores its local index data."""

    base_dir: Path

    def workspace_dir(self, ws: Workspace) -> Path:
        """Return the directory holding data for a workspace."""
        return self.base_dir / "workspaces" / ws.id

    def ensure(self, ws: Workspace) -> Path:
        """Create required directories and return workspace dir."""
        wdir = self.workspace_dir(ws)
        (wdir / "embeddings").mkdir(parents=True, exist_ok=True)
        (wdir / "docs").mkdir(parents=True, exist_ok=True)
        return wdir


def default_store_dir(local_store: bool, workspace_root: Path) -> Path:
    """Compute default storage directory.

    Args:
        local_store: If True, store index inside the workspace under `.wikdoc/`.
            If False, store index under the user's home directory `~/.wikdoc/`.
        workspace_root: Workspace path.

    Returns:
        A path to the storage root directory.
    """
    if local_store:
        return workspace_root / ".wikdoc"
    return Path.home() / ".wikdoc"


@dataclass
class IndexOptions:
    """Indexing options for scanning and chunking."""

    include_ext: List[str] = field(default_factory=lambda: list(DEFAULT_INCLUDE_EXT))
    exclude_globs: List[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_GLOBS))
    max_file_mb: float = 2.0
    follow_symlinks: bool = False
    use_gitignore: bool = True
    chunk_chars: int = 6000
    chunk_overlap_chars: int = 800


@dataclass
class RuntimeOptions:
    """Runtime options for retrieval and generation."""

    top_k: int = 8
    max_context_chars: int = 45000


@dataclass
class BackendOptions:
    """Backend selection for embeddings and LLM."""

    embedder: str = "ollama"  # "ollama" | "sbert"
    llm: str = "ollama"       # "ollama" | "llama-cpp" (future)
    model: str = "qwen2.5-coder:7b"
    embed_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
