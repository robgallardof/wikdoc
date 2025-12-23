"""Interactive terminal menu (wizard).

This is the default way to use Wikdoc.

Concepts:
  - Workspace: the folder you want to index (a project).
  - Index: the local RAG data (embeddings + chunks + manifest).
  - Pack: an exported zip of the index that can be shared/imported.

Notes:
  - This does NOT "train" the model. It builds a retrieval index (RAG).
  - Sharing a pack may leak code/documentation snippets. Treat packs as sensitive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.prompt import Confirm, IntPrompt, Prompt

from ..cli_actions import (
    do_ask,
    do_chat,
    do_docs,
    do_index,
    do_reset,
    do_status,
    do_pack_export,
    do_pack_import,
    do_workspaces_list,
)
from ..ollama_client import check_ollama
from ..config import default_store_dir
from ..integrations.openwebui import (
    detect_openwebui_cmd,
    install_openwebui,
    start_openwebui,
    open_webui_in_browser,
)


console = Console()


def _check_ollama(host: str) -> bool:
    try:
        return check_ollama(host)
    except Exception:
        return False


def _header(workspace_path: Optional[str], workspace_name: Optional[str], local_store: bool) -> None:
    console.print("")
    console.print("[bold cyan]Wikdoc[/bold cyan] — local RAG for folders")
    if workspace_path:
        mode = ".wikdoc (local)" if local_store else "~/.wikdoc (global)"
        console.print(f"[bold]Active workspace:[/bold] {workspace_name or '<unnamed>'}")
        console.print(f"[bold]Path:[/bold] {workspace_path}")
        console.print(f"[bold]Store:[/bold] {mode}")
    else:
        console.print("[dim]No active workspace set yet.[/dim]")


def _workspace_menu(workspace_path: Optional[str], workspace_name: Optional[str], local_store: bool):
    while True:
        console.print("\n[bold]Workspace[/bold]")
        console.print("1) Set workspace path")
        console.print("2) Set workspace name")
        console.print("3) Store mode (local/global)")
        console.print("4) List global workspaces")
        console.print("5) Back")

        c = Prompt.ask("Select", default="1")

        if c == "1":
            workspace_path = Prompt.ask("Enter workspace folder path")
            if workspace_path:
                local_store = Confirm.ask("Store index inside the workspace (.wikdoc)?", default=True)
            continue

        if c == "2":
            workspace_name = Prompt.ask("Friendly name for this workspace", default=workspace_name or "")
            if workspace_name == "":
                workspace_name = None
            continue

        if c == "3":
            local_store = Confirm.ask("Use local store (.wikdoc inside the workspace)?", default=local_store)
            continue

        if c == "4":
            console.print("")
            do_workspaces_list()
            console.print("[dim]Tip:[/dim] To use one of these, copy its root path and set it as workspace path.")
            continue

        if c == "5":
            return workspace_path, workspace_name, local_store

        console.print("[yellow]Invalid option.[/yellow]")


def _packs_menu(workspace_path: Optional[str], workspace_name: Optional[str], local_store: bool):
    while True:
        console.print("\n[bold]Packs (Export / Import)[/bold]")
        console.print("1) Export pack (current workspace)")
        console.print("2) Import pack (into a folder)")
        console.print("3) Back")

        c = Prompt.ask("Select", default="1")

        if c == "1":
            if not workspace_path:
                console.print("[yellow]Set an active workspace first (Workspace > Set path).[/yellow]")
                continue
            out_file = Prompt.ask(
                "Output file (will be .wikdocpack.zip)",
                default=str(Path.cwd() / ((workspace_name or "workspace") + ".wikdocpack.zip")),
            )
            do_pack_export(path=workspace_path, local_store=local_store, out_file=out_file, name=workspace_name)
            continue

        if c == "2":
            pack_file = Prompt.ask("Pack file (.wikdocpack.zip)")
            mount_path = Prompt.ask("Mount path (folder that represents the workspace on THIS machine)")
            use_local = Confirm.ask("Store imported index inside mount path (.wikdoc)?", default=True)
            overwrite = Confirm.ask("Overwrite existing index if present?", default=False)

            nm = Prompt.ask("Optional workspace name (leave blank to keep current)", default="")
            nm = nm or None

            do_pack_import(
                pack_file=pack_file,
                mount_path=mount_path,
                local_store=use_local,
                name=nm,
                overwrite=overwrite,
            )
            continue

        if c == "3":
            return

        console.print("[yellow]Invalid option.[yellow]")


def run_menu() -> None:
    """Run the interactive menu."""
    workspace_path: Optional[str] = None
    workspace_name: Optional[str] = None
    local_store: bool = True

    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen2.5-coder:7b"
    embed_model: str = "nomic-embed-text"
    top_k: int = 8

    while True:
        _header(workspace_path, workspace_name, local_store)

        console.print("\n[bold]Main Menu[/bold]")
        console.print("1) Workspace")
        console.print("2) Index workspace")
        console.print("3) Ask a question")
        console.print("4) Chat (interactive Q&A)")
        console.print("5) Generate docs (Markdown)")
        console.print("6) Packs (export/import index)")
        console.print("7) Status")
        console.print("8) Reset index")
        console.print("9) Ollama settings / connectivity")
        console.print("10) Open WebUI (install/run)")
        console.print("0) Exit")

        choice = Prompt.ask("Select", default="1")

        if choice == "1":
            workspace_path, workspace_name, local_store = _workspace_menu(workspace_path, workspace_name, local_store)
            continue

        if choice == "2":
            if not workspace_path:
                console.print("[yellow]Set a workspace path first (Main Menu > Workspace).[/yellow]")
                continue
            do_index(
                path=workspace_path,
                local_store=local_store,
                include_ext=None,
                max_file_mb=2.0,
                embedder="ollama",
                embed_model=embed_model,
                ollama_host=ollama_host,
                name=workspace_name,
            )
            continue

        if choice == "3":
            if not workspace_path:
                console.print("[yellow]Set a workspace path first.[/yellow]")
                continue

            console.print("[dim]Quick Ask: type a question and get an answer. "
                          "Press Enter on an empty prompt to return to the menu. "
                          "Type 'chat' (or 4) to open the full chat mode.[/dim]")

            while True:
                q = Prompt.ask("Question (blank to return)", default="", show_default=False)
                if not q or not q.strip():
                    break

                q_clean = q.strip().lower()
                if q_clean in ("4", "chat", "/chat"):
                    do_chat(
                        path=workspace_path,
                        local_store=local_store,
                        embedder="ollama",
                        embed_model=embed_model,
                        llm_model=llm_model,
                        ollama_host=ollama_host,
                        top_k=top_k,
                        name=workspace_name,
                    )
                    break

                do_ask(
                    path=workspace_path,
                    local_store=local_store,
                    question=q,
                    embedder="ollama",
                    embed_model=embed_model,
                    llm_model=llm_model,
                    ollama_host=ollama_host,
                    top_k=top_k,
                    name=workspace_name,
                )
            continue

        if choice == "4":
            if not workspace_path:
                console.print("[yellow]Set a workspace path first.[/yellow]")
                continue
            do_chat(
                path=workspace_path,
                local_store=local_store,
                embedder="ollama",
                embed_model=embed_model,
                llm_model=llm_model,
                ollama_host=ollama_host,
                top_k=top_k,
                name=workspace_name,
            )
            continue

        if choice == "5":
            if not workspace_path:
                console.print("[yellow]Set a workspace path first.[/yellow]")
                continue
            do_docs(
                path=workspace_path,
                local_store=local_store,
                embedder="ollama",
                embed_model=embed_model,
                llm_model=llm_model,
                ollama_host=ollama_host,
                top_k=top_k,
                name=workspace_name,
            )
            continue

        if choice == "6":
            _packs_menu(workspace_path, workspace_name, local_store)
            continue

        if choice == "7":
            if not workspace_path:
                console.print("[yellow]Set a workspace path first.[/yellow]")
                continue
            do_status(path=workspace_path, local_store=local_store)
            continue

        if choice == "8":
            if not workspace_path:
                console.print("[yellow]Set a workspace path first.[/yellow]")
                continue
            if Confirm.ask("Are you sure you want to reset the index?", default=False):
                do_reset(path=workspace_path, local_store=local_store)
            continue

        if choice == "9":
            console.print("\n[bold]Ollama Settings[/bold]")
            ollama_host = Prompt.ask("Ollama host", default=ollama_host)
            llm_model = Prompt.ask("LLM model", default=llm_model)
            embed_model = Prompt.ask("Embedding model", default=embed_model)
            top_k = IntPrompt.ask("Top-K retrieved chunks", default=top_k)

            ok = _check_ollama(ollama_host)
            if ok:
                console.print("[green]Ollama looks reachable.[/green]")
            else:
                console.print("[red]Cannot reach Ollama.[/red] Start it in another terminal: `ollama serve`.")
            continue


        if choice == "10":
            console.print("\n[bold]Open WebUI[/bold] — browser UI (optional)\n")
            status = detect_openwebui_cmd()
            console.print(f"[dim]Detection:[/dim] {status.reason}")
            console.print("")
            console.print("1) Start Open WebUI (open-webui serve)")
            console.print("2) Open WebUI in browser")
            console.print("3) Show quick instructions")
            console.print("0) Back")

            sub = Prompt.ask("Select", default="0").strip()
            if sub == "0":
                continue

            port = Prompt.ask("Port", default="8080").strip() or "8080"
            url = f"http://127.0.0.1:{port}"

            if sub == "1":
                if not status.cmd:
                    console.print("[red]Open WebUI is not installed or not found on PATH.[/red]")
                    console.print("Try: [bold]pip install -U open-webui[/bold]\n")
                    continue

                # Pick a stable directory for Open WebUI state.
                # - Local store: inside the workspace (.wikdoc/)
                # - Global store: ~/.wikdoc/
                ws_root = Path(workspace_path) if workspace_path else Path.cwd()
                store_dir = default_store_dir(local_store, ws_root)

                pid, msg = start_openwebui(
                    store_dir=store_dir,
                    ollama_base_url=ollama_host,
                    open_browser=True,
                    url=url,
                    port=int(port),
                )
                console.print(msg)
                console.print("")
                continue

            if sub == "2":
                open_webui_in_browser(url)
                console.print("[green]Opened browser.[/green]\n")
                continue

            if sub == "3":
                console.print("\n[bold]Install[/bold]\n  pip install -U open-webui")
                console.print("[bold]Run[/bold]\n  open-webui serve --port " + port)
                console.print("[bold]Open[/bold]\n  " + url + "\n")
                continue

            continue
        if choice == "0":
            console.print("Bye.")
            return

        console.print("[yellow]Invalid option.[/yellow]")