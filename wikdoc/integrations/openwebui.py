"""Open WebUI integration helpers.

Wikdoc builds a retrieval index (RAG) and offers a terminal UI.
Open WebUI is a separate project that provides a browser UI for Ollama.

This module provides best-effort helpers to:
- detect whether `open-webui` is installed
- start it and open a browser tab

Security note:
If you export/share a Wikdoc pack (index), it may contain code/document snippets.
Treat packs as sensitive.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class OpenWebUIStatus:
    """Detection result for Open WebUI."""

    cmd: Optional[str]
    reason: str


def detect_openwebui_cmd() -> OpenWebUIStatus:
    """Detect the Open WebUI CLI command on PATH."""
    for candidate in ("open-webui", "openwebui"):
        if shutil.which(candidate):
            return OpenWebUIStatus(cmd=candidate, reason=f"Found '{candidate}' on PATH.")
    return OpenWebUIStatus(cmd=None, reason="Open WebUI command not found on PATH.")


def install_openwebui() -> int:
    """Install Open WebUI via pip (best-effort).

    Returns:
        int: process return code (0 means OK)
    """
    try:
        proc = subprocess.run(
            ["python", "-m", "pip", "install", "-U", "open-webui"],
            check=False,
        )
        return int(proc.returncode)
    except Exception:
        return 1


def start_openwebui(
    store_dir: Path,
    ollama_base_url: str,
    open_browser: bool = True,
    url: str = "http://127.0.0.1:8080",
    port: int = 8080,
) -> Tuple[Optional[int], str]:
    """Start Open WebUI server in a subprocess.

    Args:
        store_dir: Directory used as the process working directory (stable place for data/logs).
        ollama_base_url: Ollama base URL (e.g. http://localhost:11434)
        open_browser: Whether to open a browser tab after starting
        url: URL to open in the browser
        port: Port for Open WebUI

    Returns:
        (pid, message)
    """
    status = detect_openwebui_cmd()
    if not status.cmd:
        return None, "Open WebUI is not installed or not found on PATH. Try: pip install -U open-webui"

    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OLLAMA_BASE_URL"] = ollama_base_url

    cmd = [status.cmd, "serve", "--host", "127.0.0.1", "--port", str(port)]
    try:
        proc = subprocess.Popen(cmd, cwd=str(store_dir), env=env)
    except Exception as e:
        return None, f"Failed to start Open WebUI: {e}"

    # Give it a moment to bind before opening the browser.
    time.sleep(1.2)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    return int(proc.pid), f"Started Open WebUI (pid={proc.pid}). If it didn't open, visit: {url}"


def open_webui_in_browser(url: str = "http://127.0.0.1:8080") -> None:
    """Open the Open WebUI URL in your default browser."""
    webbrowser.open(url)
