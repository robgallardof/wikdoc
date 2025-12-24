"""Wikdoc CLI.

Commands:
  - index: scan a workspace folder and build/update embeddings index
  - ask: ask a question and get an answer with citations
  - chat: interactive ask loop
  - docs: generate Markdown documentation outputs
  - status: show index stats
  - reset: delete index
  - wizard: interactive menu

Ollama:
  - API base is served by default at http://localhost:11434/api
  - Chat endpoint: /api/chat
  - Embeddings endpoint: /api/embeddings
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .cli_actions import do_ask, do_docs, do_index, do_reset, do_status
from .ui.menu import run_menu

app = typer.Typer(add_completion=False, help="Wikdoc: index any folder, ask questions, generate docs.")


@app.command()
def index(
    path: str = typer.Argument(..., help="Workspace folder path to index."),
    name: Optional[str] = typer.Option(None, "--name", help="Optional friendly workspace name."),
    local_store: bool = typer.Option(False, "--local-store", help="Store index inside workspace under .wikdoc/"),
    include_ext: Optional[str] = typer.Option(None, "--include-ext", help="Comma-separated extensions to include."),
    exclude_globs: Optional[str] = typer.Option(None, "--exclude-globs", help="Comma-separated glob patterns to ignore."),
    max_file_mb: Optional[float] = typer.Option(None, "--max-file-mb", help="Max file size to index (MB)."),
    embedder: str = typer.Option("ollama", "--embedder", help="Embedding backend: ollama|sbert"),
    embed_model: str = typer.Option("nomic-embed-text", "--embed-model", help="Ollama embedding model name."),
    ollama_host: str = typer.Option("http://localhost:11434", "--ollama-host", help="Ollama host URL."),
):
    """Index or update a workspace folder."""
    do_index(
        path=path,
        name=name,
        local_store=local_store,
        include_ext=include_ext,
        exclude_globs=exclude_globs,
        max_file_mb=max_file_mb,
        embedder=embedder,
        embed_model=embed_model,
        ollama_host=ollama_host,
    )


@app.command()
def ask(
    path: str = typer.Argument(..., help="Workspace folder path (must be indexed first)."),
    question: str = typer.Argument(..., help="Question to ask."),
    name: Optional[str] = typer.Option(None, "--name", help="Optional workspace name (cosmetic)."),
    local_store: bool = typer.Option(False, "--local-store", help="Use local .wikdoc/ store inside workspace."),
    top_k: int = typer.Option(8, "--top-k", help="How many chunks to retrieve."),
    llm_model: str = typer.Option("qwen2.5-coder:7b", "--model", help="Ollama LLM model."),
    embed_model: str = typer.Option("nomic-embed-text", "--embed-model", help="Ollama embedding model."),
    ollama_host: str = typer.Option("http://localhost:11434", "--ollama-host", help="Ollama host URL."),
):
    """Ask a question about an indexed workspace and get citations."""
    do_ask(
        path=path,
        question=question,
        name=name,
        local_store=local_store,
        top_k=top_k,
        llm_model=llm_model,
        embed_model=embed_model,
        ollama_host=ollama_host,
    )


@app.command()
def chat(
    path: str = typer.Argument(..., help="Workspace folder path."),
    local_store: bool = typer.Option(False, "--local-store", help="Use local .wikdoc/ store inside workspace."),
    llm_model: str = typer.Option("qwen2.5-coder:7b", "--model", help="Ollama LLM model."),
    embed_model: str = typer.Option("nomic-embed-text", "--embed-model", help="Ollama embedding model."),
    ollama_host: str = typer.Option("http://localhost:11434", "--ollama-host", help="Ollama host URL."),
    top_k: int = typer.Option(8, "--top-k"),
):
    """Interactive chat loop."""
    while True:
        q = typer.prompt("wikdoc>").strip()
        if q.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        if not q:
            continue
        do_ask(
            path=path,
            question=q,
            local_store=local_store,
            top_k=top_k,
            llm_model=llm_model,
            embed_model=embed_model,
            ollama_host=ollama_host,
        )


@app.command()
def docs(
    path: str = typer.Argument(..., help="Workspace folder path."),
    out: str = typer.Option("./docs", "--out", help="Output directory for docs."),
    template: str = typer.Option("wiki", "--template", help="wiki|readme|architecture"),
    local_store: bool = typer.Option(False, "--local-store", help="Use local .wikdoc/ store inside workspace."),
    top_k: int = typer.Option(8, "--top-k", help="How many chunks to retrieve for docs."),
    llm_model: str = typer.Option("qwen2.5-coder:7b", "--model", help="Ollama LLM model."),
    embed_model: str = typer.Option("nomic-embed-text", "--embed-model", help="Ollama embedding model."),
    ollama_host: str = typer.Option("http://localhost:11434", "--ollama-host", help="Ollama host URL."),
):
    """Generate Markdown documentation from an indexed workspace."""
    do_docs(
        path=path,
        out=out,
        template=template,
        local_store=local_store,
        top_k=top_k,
        llm_model=llm_model,
        embed_model=embed_model,
        ollama_host=ollama_host,
    )


@app.command()
def status(
    path: str = typer.Argument(..., help="Workspace folder path."),
    local_store: bool = typer.Option(False, "--local-store", help="Use local .wikdoc/ store inside workspace."),
):
    """Show index stats for a workspace."""
    do_status(path=path, local_store=local_store)


@app.command()
def reset(
    path: str = typer.Argument(..., help="Workspace folder path."),
    local_store: bool = typer.Option(False, "--local-store", help="Use local .wikdoc/ store inside workspace."),
):
    """Reset (delete) index data for a workspace."""
    do_reset(path=path, local_store=local_store)




@app.command("workspaces")
def workspaces_list():
    """List workspaces stored in the global store (~/.wikdoc)."""
    do_workspaces_list()


@app.command("export-pack")
def export_pack_cmd(
    path: str = typer.Argument(..., help="Workspace folder path (must be indexed)."),
    out_file: str = typer.Argument(..., help="Output .wikdocpack.zip file path."),
    name: Optional[str] = typer.Option(None, "--name", help="Optional workspace name."),
    local_store: bool = typer.Option(True, "--local-store/--global-store", help="Use local (.wikdoc) store or global (~/.wikdoc)."),
):
    """Export a portable pack for sharing."""
    do_pack_export(path=path, local_store=local_store, out_file=out_file, name=name)


@app.command("import-pack")
def import_pack_cmd(
    pack_file: str = typer.Argument(..., help="Path to a .wikdocpack.zip file."),
    mount_path: str = typer.Argument(..., help="Workspace folder on this machine."),
    name: Optional[str] = typer.Option(None, "--name", help="Optional workspace name."),
    local_store: bool = typer.Option(True, "--local-store/--global-store", help="Import into local (.wikdoc) store or global store."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite if an index already exists."),
):
    """Import a pack into this machine."""
    do_pack_import(pack_file=pack_file, mount_path=mount_path, local_store=local_store, name=name, overwrite=overwrite)

@app.command()
def wizard():
    """Launch an interactive terminal wizard (menu)."""
    run_menu()


@app.command()
def start():
    """Alias for `wizard` (interactive terminal menu)."""
    run_menu()


@app.command()
def webui(
    path: str = typer.Argument(..., help="Workspace folder path (must be indexed)."),
    local_store: bool = typer.Option(False, "--local-store", help="Use the index inside <path>/.wikdoc."),
    llm_model: str = typer.Option("qwen2.5-coder:7b", "--model", help="Ollama LLM model."),
    embed_model: str = typer.Option("nomic-embed-text", "--embed-model", help="Ollama embedding model."),
    ollama_host: str = typer.Option("http://localhost:11434", "--ollama-host", help="Ollama host URL."),
    top_k: int = typer.Option(8, "--top-k", help="How many chunks to retrieve."),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind the web UI server."),
    port: int = typer.Option(7860, "--port", help="Port to bind the web UI server."),
    name: Optional[str] = typer.Option(None, "--name", help="Optional workspace name (cosmetic)."),
    open_browser: bool = typer.Option(
        False,
        "--browser/--no-browser",
        help="Open the Web UI in your default browser when launching.",
    ),
):
    """Start a lightweight browser UI for asking questions."""

    from .ui.webui import launch_webui

    launch_webui(
        path=path,
        local_store=local_store,
        top_k=top_k,
        llm_model=llm_model,
        embed_model=embed_model,
        ollama_host=ollama_host,
        host=host,
        port=port,
        name=name,
        open_browser=open_browser,
    )



@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 17863,
    ollama_host: str = "http://localhost:11434",
    llm_model: str = "qwen2.5-coder:7b",
    embed_model: str = "nomic-embed-text",
    top_k: int = 8,
    local_store: bool = typer.Option(
        False,
        "--local-store",
        help="Serve the local store at --path/.wikdoc (instead of the global store at ~/.wikdoc).",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Workspace root folder. Required when using --local-store.",
    ),
    store_dir: Optional[Path] = typer.Option(
        None,
        "--store-dir",
        help="Override the base store directory (advanced).",
    ),
):
    """Start the local Wikdoc RAG API (OpenAI-compatible).

    Any OpenAI-compatible client can connect to this endpoint:
      Base URL: http://127.0.0.1:<port>/v1
      API key: any value (not used)
    """
    import uvicorn
    from .server.rag_api import create_app

    if local_store and not path:
        raise typer.BadParameter("--path is required when using --local-store")

    app = create_app(
        ollama_host=ollama_host,
        llm_model=llm_model,
        embed_model=embed_model,
        top_k=top_k,
        local_store=local_store,
        workspace_path=str(path) if path else None,
        store_dir=str(store_dir) if store_dir else None,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
