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
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import requests

OPENWEBUI_VERSION = "0.6.43"


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


def check_rag_api(host: str, port: int, timeout: float = 0.8) -> Tuple[bool, str]:
    """Check whether the Wikdoc RAG API is reachable."""
    base = f"http://{host}:{port}"
    url = f"{base}/v1"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return False, f"API responded {r.status_code} at {url}."
        models_url = f"{base}/v1/models"
        r2 = requests.get(models_url, timeout=timeout)
        if r2.ok:
            json_data = r2.json()
            data = json_data if isinstance(json_data, dict) else {}
            count = len(data.get("data", []))
            return True, f"Up at {url} (models: {count})."
        return True, f"Up at {url} (models check: {r2.status_code})."
    except requests.RequestException as exc:
        return False, f"Not reachable at {url} ({exc.__class__.__name__})."


def install_openwebui() -> int:
    """Install Open WebUI via pip (best-effort).

    Returns:
        int: process return code (0 means OK)
    """
    try:
        proc = subprocess.run(
            ["python", "-m", "pip", "install", "-U", f"open-webui=={OPENWEBUI_VERSION}"],
            check=False,
        )
        return int(proc.returncode)
    except Exception:
        return 1


def _is_port_open(host: str, port: int) -> bool:
    """Check whether a TCP port is already accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def _wikdoc_base_cmd() -> list[str]:
    """Return the best command to invoke the Wikdoc CLI."""
    if shutil.which("wikdoc"):
        return ["wikdoc"]
    return [sys.executable, "-m", "wikdoc"]


def start_rag_api(
    host: str,
    port: int,
    ollama_host: str,
    llm_model: str,
    embed_model: str,
    top_k: int,
    local_store: bool,
    workspace_path: Optional[Path],
    store_dir: Optional[Path] = None,
) -> Tuple[Optional[int], str]:
    """Start the Wikdoc RAG API (OpenAI-compatible) in a subprocess."""
    if _is_port_open(host, port):
        return None, f"Wikdoc API already running at http://{host}:{port}/v1."

    cmd = _wikdoc_base_cmd() + [
        "serve",
        "--host",
        host,
        "--port",
        str(port),
        "--ollama-host",
        ollama_host,
        "--model",
        llm_model,
        "--embed-model",
        embed_model,
        "--top-k",
        str(top_k),
    ]
    if local_store:
        cmd.append("--local-store")
        if workspace_path:
            cmd.extend(["--path", str(workspace_path)])
    if store_dir:
        cmd.extend(["--store-dir", str(store_dir)])

    try:
        proc = subprocess.Popen(cmd)
    except Exception as exc:
        return None, f"Failed to start Wikdoc API: {exc}"

    return int(proc.pid), f"Started Wikdoc API (pid={proc.pid}) at http://{host}:{port}/v1."


def start_openwebui(
    store_dir: Path,
    ollama_base_url: str,
    openai_api_base: Optional[str] = None,
    open_browser: bool = True,
    url: str = "http://127.0.0.1:8080",
    port: int = 8080,
) -> Tuple[Optional[int], str]:
    """Start Open WebUI server in a subprocess.

    Args:
        store_dir: Directory used as the process working directory (stable place for data/logs).
        ollama_base_url: Ollama base URL (e.g. http://localhost:11434)
        openai_api_base: Optional OpenAI-compatible API base URL for Wikdoc RAG.
        open_browser: Whether to open a browser tab after starting.
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
    if openai_api_base:
        env["OPENAI_API_BASE_URL"] = openai_api_base
        env["OPENAI_API_BASE"] = openai_api_base
        env.setdefault("OPENAI_API_KEY", "sk-local")

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
