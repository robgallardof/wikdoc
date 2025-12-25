# wikdoc/cli_actions.py
"""
Reusable CLI actions.

The main CLI (`wikdoc.cli`) calls these functions, and the wizard menu
(`wikdoc.ui.menu`) reuses them.

This keeps the UI thin and makes behaviors testable.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional, Any

import typer
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.prompt import Prompt

from .config import (
    BackendOptions,
    IndexOptions,
    RuntimeOptions,
    StoreLayout,
    Workspace,
    load_index_options,
    default_store_dir,
)
from .packs import (
    export_pack,
    import_pack,
    write_workspace_json,
    list_global_workspaces,
)
from .chunking.base import Chunk
from .chunking.fallback import FallbackChunker
from .embeddings.ollama import OllamaEmbedder
from .embeddings.sbert import SentenceTransformersEmbedder
from .ingest.scanner import Document, list_candidate_files, scan_files
from .vectordb.sqlite_numpy import SQLiteNumpyVectorStore
from .rag.retrieve import retrieve
from .rag.prompt import build_prompt
from .rag.answer import OllamaLLM

console = Console()


def _resolve_store(path: Path, local_store: bool) -> Path:
    """
    Resolve the base store directory for a workspace.

    Args:
        path: Workspace root path.
        local_store: If True, store is under <workspace>/.wikdoc. Else under ~/.wikdoc.

    Returns:
        Store base directory.
    """
    return default_store_dir(local_store=local_store, workspace_root=path)


def _workspace_and_layout(path: str, name: Optional[str], local_store: bool) -> tuple[Workspace, StoreLayout, Path]:
    """
    Build a workspace model and its store layout.

    Args:
        path: Workspace root path (string).
        name: Optional friendly name.
        local_store: Storage mode.

    Returns:
        (workspace, layout, workspace_store_dir)
    """
    ws = Workspace.from_path(path, name=name)
    store_root = _resolve_store(ws.root, local_store=local_store)
    layout = StoreLayout(base_dir=store_root)
    wdir = layout.ensure(ws)
    return ws, layout, wdir


def _make_embedder(embedder: str, ollama_host: str, embed_model: str):
    """
    Create an embedding backend from CLI options.

    Args:
        embedder: "ollama" or "sbert".
        ollama_host: Ollama base URL.
        embed_model: Ollama embedding model name.

    Returns:
        Embedder instance.

    Raises:
        typer.BadParameter: If embedder is unknown.
    """
    if embedder == "ollama":
        return OllamaEmbedder(host=ollama_host, model=embed_model)
    if embedder == "sbert":
        return SentenceTransformersEmbedder()
    raise typer.BadParameter(f"Unknown embedder: {embedder}")


def _flush_embedding_batch(
    embedder: Any,
    store: SQLiteNumpyVectorStore,
    batch_texts: list[str],
    batch_meta: list[tuple[Document, Chunk]],
) -> int:
    """
    Embed the current batch and persist it to the vector store.

    Args:
        embedder: Embedder instance to generate vectors.
        store: Vector store for persistence.
        batch_texts: Raw chunk text list, aligned to batch_meta.
        batch_meta: Document/chunk pairs aligned to batch_texts.

    Returns:
        The number of chunks written by the store.

    Raises:
        Exception: Propagates embedding backend failures with context printed to console.
    """
    if not batch_texts:
        return 0

    try:
        vecs = embedder.embed(batch_texts)
    except Exception:
        failing = batch_meta[0][0].rel_path if batch_meta else "<unknown>"
        console.print("")
        console.print("[red]Embedding backend error.[/red]")
        console.print(f"[yellow]While processing:[/yellow] {failing}")
        console.print("[yellow]Fix options:[/yellow] exclude lockfiles/minified files, or lower WIKDOC_EMBED_MAX_CHARS.")
        raise

    payload = []
    for (doc, chunk), vec in zip(batch_meta, vecs):
        payload.append(
            {
                "file_path": doc.rel_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": doc.language,
                "symbol": chunk.symbol,
                "file_sha256": doc.sha256,
                "text": chunk.text,
                "embedding": vec,
            }
        )

    batch_texts.clear()
    batch_meta.clear()
    return store.upsert_chunks(payload)


def do_index(
    path: str,
    local_store: bool,
    include_ext: Optional[str],
    exclude_globs: Optional[str],
    max_file_mb: Optional[float],
    embedder: str,
    embed_model: str,
    ollama_host: str,
    name: Optional[str] = None,
) -> None:
    """
    Index or update a workspace folder with progress bars.

    Args:
        path: Workspace root folder path.
        local_store: Store in <workspace>/.wikdoc if True, else ~/.wikdoc.
        include_ext: Comma-separated extensions to include.
        exclude_globs: Comma-separated ignore globs.
        max_file_mb: Maximum file size (MB) to index.
        embedder: Embedding backend ("ollama" or "sbert").
        embed_model: Ollama embedding model (when embedder=ollama).
        ollama_host: Ollama base URL.
        name: Optional friendly workspace name.
    """
    ws, layout, wdir = _workspace_and_layout(path, name, local_store)
    backend = BackendOptions(embedder=embedder, llm="ollama", model="", embed_model=embed_model, ollama_host=ollama_host)

    # Persist workspace metadata (helps listing + pack portability)
    try:
        write_workspace_json(wdir, ws, backend)
    except Exception:
        pass

    idx_opts: IndexOptions = load_index_options(ws.root)
    if max_file_mb is not None:
        idx_opts.max_file_mb = max_file_mb
    if include_ext:
        idx_opts.include_ext = [x.strip().lstrip(".") for x in include_ext.split(",") if x.strip()]
    if exclude_globs:
        idx_opts.exclude_globs = [x.strip() for x in exclude_globs.split(",") if x.strip()]

    embed = _make_embedder(embedder, ollama_host, embed_model)
    chunker = FallbackChunker(chunk_chars=idx_opts.chunk_chars, overlap_chars=idx_opts.chunk_overlap_chars)
    store = SQLiteNumpyVectorStore(store_dir=wdir)

    manifest_path = wdir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}

    # Pre-list candidates for progress total
    console.print("[dim]Scanning workspace for candidate files...[/dim]")
    files = list_candidate_files(ws.root, idx_opts)

    batch_size = 32
    batch_texts: list[str] = []
    batch_meta: list[tuple[Document, Chunk]] = []

    total_files = len(files)
    changed_files = 0
    written_chunks = 0

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Indexing files", total=total_files)

        for doc in scan_files(ws.root, files, idx_opts):
            progress.advance(task, 1)

            prev = manifest.get(doc.rel_path)
            if prev and prev.get("sha256") == doc.sha256:
                continue  # unchanged

            changed_files += 1
            chunks = chunker.chunk(doc.content)
            if not chunks:
                manifest[doc.rel_path] = {"sha256": doc.sha256}
                continue

            for ch in chunks:
                batch_texts.append(ch.text)
                batch_meta.append((doc, ch))

                if len(batch_texts) >= batch_size:
                    written_chunks += _flush_embedding_batch(embed, store, batch_texts, batch_meta)

            manifest[doc.rel_path] = {"sha256": doc.sha256}

    written_chunks += _flush_embedding_batch(embed, store, batch_texts, batch_meta)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    stats = store.stats()
    console.print(f"\n[bold green]Indexed workspace:[/bold green] {ws.root}")
    console.print(f"Store: {wdir}")
    console.print(f"Candidate files: {total_files}")
    console.print(f"Changed files: {changed_files}")
    console.print(f"Chunks written (this run): {written_chunks}")
    console.print(f"Total chunks in index: {stats.get('chunks')}")
    store.close()


def do_ask(
    path: str,
    question: str,
    local_store: bool,
    top_k: int,
    llm_model: str,
    embed_model: str,
    ollama_host: str,
    embedder: str = "ollama",
    name: Optional[str] = None,
) -> None:
    """
    Ask a question about an indexed workspace and show citations.

    Args:
        path: Workspace root path.
        question: User question.
        local_store: Storage mode.
        top_k: Number of chunks to retrieve.
        llm_model: Ollama chat model.
        embed_model: Embedding model.
        ollama_host: Ollama base URL.
        embedder: "ollama" or "sbert".
        name: Optional workspace name.
    """
    ws, layout, wdir = _workspace_and_layout(path, name, local_store)
    backend = BackendOptions(embedder=embedder, llm="ollama", model=llm_model, embed_model=embed_model, ollama_host=ollama_host)

    # Persist workspace metadata (helps listing + pack portability)
    try:
        write_workspace_json(wdir, ws, backend)
    except Exception:
        pass

    store = SQLiteNumpyVectorStore(store_dir=wdir)

    # ✅ Use configured embedder (bugfix: previously forced OllamaEmbedder)
    embed = _make_embedder(embedder, ollama_host, embed_model)
    llm = OllamaLLM(host=ollama_host, model=llm_model)
    runtime = RuntimeOptions(top_k=top_k)

    qvec = embed.embed([question])[0]
    if not qvec:
        console.print("[red]Query embedding is empty.[/red]")
        console.print("[yellow]Check your embedding backend and Ollama embedding endpoint/response.[/yellow]")
        store.close()
        return

    hits = retrieve(store, qvec, top_k=runtime.top_k)

    if not hits:
        console.print("[yellow]No relevant context found in the index.[/yellow]")
        store.close()
        return

    messages = build_prompt(question=question, hits=hits, max_context_chars=runtime.max_context_chars)

    with Progress(SpinnerColumn(), TextColumn("Thinking..."), TimeElapsedColumn(), console=console) as p:
        t = p.add_task("thinking", total=None)
        answer = llm.chat(messages)
        p.update(t, completed=1)

    console.print("\n[bold]Answer[/bold]\n")
    console.print(answer)

    console.print("\n[dim]Top sources[/dim]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Source")
    for h in hits[: min(10, len(hits))]:
        table.add_row(f"{h.score:.3f}", f"{h.path}:{h.start_line}-{h.end_line}")
    console.print(table)
    store.close()


def do_chat(
    path: str,
    local_store: bool,
    top_k: int,
    llm_model: str,
    embed_model: str,
    ollama_host: str,
    embedder: str = "ollama",
    name: Optional[str] = None,
) -> None:
    """
    Start an interactive Q&A loop over an indexed workspace.

    Menu-friendly equivalent of the `wikdoc chat` command.

    Commands:
      - /exit : leave chat
      - /help : show help
    """
    console.print("")
    console.print("[bold]Chat mode[/bold] — ask questions about your workspace.")
    console.print("Commands: [cyan]/exit[/cyan], [cyan]/help[/cyan]")
    console.print("")

    while True:
        q = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        if not q:
            continue
        if q.lower() in {"/exit", "exit", "quit", "/quit"}:
            console.print("[green]Leaving chat.[/green]")
            return
        if q.lower() in {"/help", "help"}:
            console.print(" - Ask anything about the code/docs you've indexed.")
            console.print(" - /exit to leave.")
            continue

        do_ask(
            path=path,
            question=q,
            name=name,
            local_store=local_store,
            top_k=top_k,
            llm_model=llm_model,
            embed_model=embed_model,
            ollama_host=ollama_host,
            embedder=embedder,
        )
        console.print("")


def do_docs(
    path: str,
    local_store: bool,
    top_k: int = 8,
    llm_model: str = "qwen2.5-coder:7b",
    embed_model: str = "nomic-embed-text",
    ollama_host: str = "http://localhost:11434",
    embedder: str = "ollama",
    out: str = "./docs",
    template: str = "wiki",
    name: Optional[str] = None,
) -> None:
    """
    Generate Markdown documentation for a workspace using RAG + an Ollama LLM.

    Templates:
      - wiki:         multiple pages (Overview, Architecture, Key Modules)
      - readme:       single README.md
      - architecture: focused architecture doc
    """
    # NOTE: This CLI action is superseded by docsgen.generate.generate_docs in some builds.
    # If you keep this richer version, keep it consistent with the vector store/search hit types.
    from .docsgen.generate import generate_docs  # local import to avoid circulars

    ws, layout, wdir = _workspace_and_layout(path, name=name, local_store=local_store)

    backend = BackendOptions(embedder=embedder, llm="ollama", model=llm_model, embed_model=embed_model, ollama_host=ollama_host)
    try:
        write_workspace_json(wdir, ws, backend)
    except Exception:
        pass

    store = SQLiteNumpyVectorStore(store_dir=wdir)
    out_dir = Path(out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # If you want LLM-enriched docs, wire it here. For now, generate skeletons.
    written = generate_docs(workspace=ws, store=store, out_dir=out_dir, template=template, llm=None, runtime=RuntimeOptions(top_k=top_k))
    console.print(f"[bold green]Generated {len(written)} file(s) into[/bold green] {out_dir}")
    for p in written:
        console.print(f" - {p}")
    store.close()


def do_status(path: str, local_store: bool) -> None:
    """
    Show vector index statistics for a workspace.

    Args:
        path: Workspace root path.
        local_store: Storage mode.
    """
    ws, layout, wdir = _workspace_and_layout(path, name=None, local_store=local_store)
    store = SQLiteNumpyVectorStore(store_dir=wdir)
    stats = store.stats()
    console.print(f"Workspace: {ws.root}")
    for k, v in stats.items():
        console.print(f"- {k}: {v}")
    store.close()


def do_reset(path: str, local_store: bool) -> None:
    """
    Delete index data and manifest for a workspace.

    Args:
        path: Workspace root.
        local_store: Storage mode.
    """
    ws, layout, wdir = _workspace_and_layout(path, name=None, local_store=local_store)
    store = SQLiteNumpyVectorStore(store_dir=wdir)
    store.reset()

    mp = wdir / "manifest.json"
    if mp.exists():
        try:
            mp.unlink()
        except Exception:
            pass

    console.print(f"[bold yellow]Index reset for workspace[/bold yellow] {ws.root}")
    store.close()


def do_pack_export(path: str, local_store: bool, out_file: str, name: Optional[str] = None) -> None:
    """
    Export current workspace index as a portable pack.

    Args:
        path: Workspace root.
        local_store: Storage mode used to locate the index.
        out_file: Output pack filename.
        name: Optional workspace name.
    """
    out_path = export_pack(path=path, local_store=local_store, out_file=out_file, name=name, include_docs=True)
    console.print(f"[green]Pack exported:[/green] {out_path}")


def do_pack_import(
    pack_file: str,
    mount_path: str,
    local_store: bool,
    name: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Import a pack into the given mount path.

    Args:
        pack_file: .wikdocpack.zip path.
        mount_path: Workspace root on this machine.
        local_store: Where to import (local vs global store).
        name: Optional name.
        overwrite: Overwrite existing store data.
    """
    wdir = import_pack(pack_file=pack_file, mount_path=mount_path, local_store=local_store, name=name, overwrite=overwrite)
    console.print(f"[green]Pack imported into:[/green] {wdir}")


def do_workspaces_list() -> None:
    """
    List workspaces in the global store (~/.wikdoc).
    """
    items = list_global_workspaces()
    if not items:
        console.print("[yellow]No global workspaces found (~/.wikdoc).[/yellow]")
        return
    console.print("[bold]Global Workspaces[/bold]")
    for i, it in enumerate(items, 1):
        nm = it.get("name") or "<unnamed>"
        root = it.get("root") or "<unknown>"
        wid = it.get("workspace_id") or "<id?>"
        console.print(f"{i}) {nm}  [dim]{root}[/dim]  (id={wid})")
