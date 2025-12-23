"""Store helpers (compat layer).

This package exists to keep imports stable (e.g. server.rag_api).
Actual implementations live in wikdoc.config.
"""

from .layout import StoreLayout, default_store_dir
from .workspace import Workspace

__all__ = ["StoreLayout", "default_store_dir", "Workspace"]
