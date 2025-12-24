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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console
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
from ..packs import list_global_workspaces


console = Console()


@dataclass
class MenuState:
    """In-memory session configuration for the wizard."""

    workspace_path: Optional[str] = None
    local_store: bool = True
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "qwen2.5-coder:7b"
    embed_model: str = "nomic-embed-text"
    top_k: int = 8


def _check_ollama(host: str) -> bool:
    try:
        return check_ollama(host)
    except Exception:
        return False


def _header(state: MenuState) -> None:
    console.print("")
    console.print("[bold cyan]Wikdoc[/bold cyan] — local RAG for folders")
    if state.workspace_path:
        mode = ".wikdoc (local)" if state.local_store else "~/.wikdoc (global)"
        console.print(f"[bold]Active workspace:[/bold] {state.workspace_path}")
        console.print(f"[bold]Store:[/bold] {mode}")
    else:
        console.print("[dim]No active workspace set yet.[/dim]")

    console.print(
        f"[dim]Ollama:[/dim] {state.ollama_host} | "
        f"[dim]Model:[/dim] {state.llm_model} | "
        f"[dim]Embed:[/dim] {state.embed_model}"
    )


def _require_workspace(state: MenuState, action: str) -> bool:
    """Ensure a workspace is configured before running an action."""
    if state.workspace_path:
        return True
    console.print(f"[yellow]Set a workspace path first to {action}.[/yellow]")
    return False


def _select_global_workspace() -> Optional[str]:
    items = list_global_workspaces()
    if not items:
        console.print("[yellow]No global workspaces found (~/.wikdoc).[/yellow]")
        return None
    console.print("\n[bold]Global Workspaces[/bold]")
    for i, it in enumerate(items, 1):
        root = it.get("root") or "<unknown>"
        wid = it.get("workspace_id") or "<id?>"
        console.print(f"{i}) [dim]{root}[/dim]  (id={wid})")
    choice = IntPrompt.ask("Pick a workspace number (0 to cancel)", default=0)
    if choice <= 0 or choice > len(items):
        return None
    return items[choice - 1].get("root")


def _workspace_menu(workspace_path: Optional[str], local_store: bool):
    while True:
        console.print("\n[bold]Workspace (namespace)[/bold]")
        console.print("1) Set workspace path (namespace)")
        console.print("2) Store mode (local/global)")
        console.print("3) Select from global workspaces")
        console.print("4) List global workspaces")
        console.print("5) Back")

        c = Prompt.ask("Select", default="1")

        if c == "1":
            workspace_path = Prompt.ask("Enter workspace folder path (namespace)")
            if workspace_path:
                local_store = Confirm.ask("Store index inside the workspace (.wikdoc)?", default=True)
            continue

        if c == "2":
            local_store = Confirm.ask("Use local store (.wikdoc inside the workspace)?", default=local_store)
            continue

        if c == "3":
            picked = _select_global_workspace()
            if picked:
                workspace_path = picked
                local_store = False
                console.print(f"[green]Selected:[/green] {workspace_path}")
            continue

        if c == "4":
            console.print("")
            do_workspaces_list()
            console.print("[dim]Tip:[/dim] To use one of these, copy its root path and set it as workspace path.")
            continue

        if c == "5":
            return workspace_path, local_store

        console.print("[yellow]Invalid option.[/yellow]")


def _packs_menu(workspace_path: Optional[str], local_store: bool):
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
            default_name = Path(workspace_path).name or "workspace"
            out_file = Prompt.ask(
                "Output file (will be .wikdocpack.zip)",
                default=str(Path.cwd() / (default_name + ".wikdocpack.zip")),
            )
            do_pack_export(path=workspace_path, local_store=local_store, out_file=out_file, name=None)
            continue

        if c == "2":
            pack_file = Prompt.ask("Pack file (.wikdocpack.zip)")
            default_mount = workspace_path or str(Path.cwd())
            mount_path = Prompt.ask(
                "Mount path (folder that represents the workspace on THIS machine)",
                default=default_mount,
            )
            use_local = Confirm.ask("Store imported index inside mount path (.wikdoc)?", default=True)
            overwrite = Confirm.ask("Overwrite existing index if present?", default=False)

            do_pack_import(
                pack_file=pack_file,
                mount_path=mount_path,
                local_store=use_local,
                name=None,
                overwrite=overwrite,
            )
            continue

        if c == "3":
            return

        console.print("[yellow]Invalid option.[yellow]")


def run_menu() -> None:
    """Run the interactive menu."""
    state = MenuState()

    while True:
        _header(state)

        console.print("\n[bold]Main Menu[/bold]")
        console.print("1) Workspace")
        console.print("2) Index workspace")
        console.print("3) Ask a question")
        console.print("4) Chat (interactive Q&A)")
        console.print("5) Generate docs (Markdown)")
        console.print("6) Packs (export/import index)")
        console.print("7) Status")
        console.print("8) Reset index")
        console.print("9) Settings (Ollama, retrieval)")
        console.print("0) Exit")

        choice = Prompt.ask("Select", default="1")

        if choice == "1":
            state.workspace_path, state.local_store = _workspace_menu(
                state.workspace_path,
                state.local_store,
            )
            continue

        if choice == "2":
            if not _require_workspace(state, "index"):
                continue
            do_index(
                path=state.workspace_path,
                local_store=state.local_store,
                include_ext=None,
                exclude_globs=None,
                max_file_mb=None,
                embedder="ollama",
                embed_model=state.embed_model,
                ollama_host=state.ollama_host,
                name=None,
            )
            continue

        if choice == "3":
            if not _require_workspace(state, "ask questions"):
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
                        path=state.workspace_path,
                        local_store=state.local_store,
                        embedder="ollama",
                    embed_model=state.embed_model,
                    llm_model=state.llm_model,
                    ollama_host=state.ollama_host,
                    top_k=state.top_k,
                    name=None,
                )
                    break

                do_ask(
                    path=state.workspace_path,
                    local_store=state.local_store,
                    question=q,
                    embedder="ollama",
                    embed_model=state.embed_model,
                    llm_model=state.llm_model,
                    ollama_host=state.ollama_host,
                    top_k=state.top_k,
                    name=None,
                )
            continue

        if choice == "4":
            if not _require_workspace(state, "start chat"):
                continue
            do_chat(
                path=state.workspace_path,
                local_store=state.local_store,
                embedder="ollama",
                embed_model=state.embed_model,
                llm_model=state.llm_model,
                ollama_host=state.ollama_host,
                top_k=state.top_k,
                name=None,
            )
            continue

        if choice == "5":
            if not _require_workspace(state, "generate docs"):
                continue
            do_docs(
                path=state.workspace_path,
                local_store=state.local_store,
                embedder="ollama",
                embed_model=state.embed_model,
                llm_model=state.llm_model,
                ollama_host=state.ollama_host,
                top_k=state.top_k,
                name=None,
            )
            continue

        if choice == "6":
            _packs_menu(state.workspace_path, state.local_store)
            continue

        if choice == "7":
            if not _require_workspace(state, "view status"):
                continue
            do_status(path=state.workspace_path, local_store=state.local_store)
            continue

        if choice == "8":
            if not _require_workspace(state, "reset the index"):
                continue
            if Confirm.ask("Are you sure you want to reset the index?", default=False):
                do_reset(path=state.workspace_path, local_store=state.local_store)
            continue

        if choice == "9":
            console.print("\n[bold]Settings[/bold]")
            state.ollama_host = Prompt.ask("Ollama host", default=state.ollama_host)
            state.llm_model = Prompt.ask("LLM model", default=state.llm_model)
            state.embed_model = Prompt.ask("Embedding model", default=state.embed_model)
            state.top_k = IntPrompt.ask("Top-K retrieved chunks", default=state.top_k)

            ok = _check_ollama(state.ollama_host)
            if ok:
                console.print("[green]Ollama looks reachable.[/green]")
            else:
                console.print("[red]Cannot reach Ollama.[/red] Start it in another terminal: `ollama serve`.")
            continue
        if choice == "0":
            console.print("Bye.")
            return

        console.print("[yellow]Invalid option.[/yellow]")
