"""Reusable CLI actions.

The main CLI (`wikdoc.cli`) calls these functions, and the wizard menu
(`wikdoc.ui.menu`) reuses them.

This keeps the UI thin and makes behaviors testable.
"""

from __future__ import annotations

import json
import os
import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, SpinnerColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.prompt import Prompt

from .config import BackendOptions, IndexOptions, RuntimeOptions, StoreLayout, Workspace
from .packs import export_pack, import_pack, write_workspace_json, list_global_workspaces, default_store_dir
from .chunking.fallback import FallbackChunker
from .embeddings.ollama import OllamaEmbedder
from .embeddings.sbert import SentenceTransformersEmbedder
from .ingest.scanner import list_candidate_files, scan_files
from .vectordb.sqlite_numpy import SQLiteNumpyVectorStore
from .rag.retrieve import retrieve
from .rag.prompt import build_prompt
from .rag.answer import OllamaLLM
from .docsgen.generate import generate_docs

console = Console()


def _resolve_store(path: Path, local_store: bool) -> Path:
    return default_store_dir(local_store=local_store, workspace_root=path)


def _workspace_and_layout(path: str, name: Optional[str], local_store: bool):
    ws = Workspace.from_path(path, name=name)
    store_root = _resolve_store(ws.root, local_store=local_store)
    layout = StoreLayout(base_dir=store_root)
    wdir = layout.ensure(ws)
    return ws, layout, wdir


def _make_embedder(embedder: str, ollama_host: str, embed_model: str):
    if embedder == "ollama":
        return OllamaEmbedder(host=ollama_host, model=embed_model)
    if embedder == "sbert":
        return SentenceTransformersEmbedder()
    raise typer.BadParameter(f"Unknown embedder: {embedder}")


def do_index(
    path: str,
    local_store: bool,
    include_ext: Optional[str],
    max_file_mb: float,
    embedder: str,
    embed_model: str,
    ollama_host: str,
    name: Optional[str] = None,
) -> None:
    """Index or update a workspace folder with progress bars."""
    ws, layout, wdir = _workspace_and_layout(path, name, local_store)
    backend = BackendOptions(embedder=embedder, llm="ollama", model="", embed_model=embed_model, ollama_host=ollama_host)
    # Persist workspace metadata (helps listing + pack portability)
    try:
        write_workspace_json(wdir, ws, backend)
    except Exception:
        pass

    idx_opts = IndexOptions(max_file_mb=max_file_mb)
    if include_ext:
        idx_opts.include_ext = [x.strip().lstrip(".") for x in include_ext.split(",") if x.strip()]

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

    BATCH = 32
    batch_texts = []
    batch_meta = []

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

                if len(batch_texts) >= BATCH:
                    try:
                        vecs = embed.embed(batch_texts)
                    except Exception:
                        failing = batch_meta[0][0].rel_path if batch_meta else "<unknown>"
                        console.print("")
                        console.print("[red]Embedding backend error.[/red]")
                        console.print(f"[yellow]While processing:[/yellow] {failing}")
                        console.print("[yellow]Fix options:[/yellow] exclude lockfiles/minified files, or lower WIKDOC_EMBED_MAX_CHARS.")
                        raise
                    payload = []
                    for (d, c), v in zip(batch_meta, vecs):
                        payload.append(
                            {
                                "file_path": d.rel_path,
                                "start_line": c.start_line,
                                "end_line": c.end_line,
                                "language": d.language,
                                "symbol": c.symbol,
                                "file_sha256": d.sha256,
                                "text": c.text,
                                "embedding": v,
                            }
                        )
                    written_chunks += store.upsert_chunks(payload)
                    batch_texts.clear()
                    batch_meta.clear()

            manifest[doc.rel_path] = {"sha256": doc.sha256}

    if batch_texts:
        try:
            vecs = embed.embed(batch_texts)
        except Exception:
            failing = batch_meta[0][0].rel_path if batch_meta else "<unknown>"
            console.print("")
            console.print("[red]Embedding backend error.[/red]")
            console.print(f"[yellow]While processing:[/yellow] {failing}")
            console.print("[yellow]Fix options:[/yellow] exclude lockfiles/minified files, or lower WIKDOC_EMBED_MAX_CHARS.")
            raise
        payload = []
        for (d, c), v in zip(batch_meta, vecs):
            payload.append(
                {
                    "file_path": d.rel_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "language": d.language,
                    "symbol": c.symbol,
                    "file_sha256": d.sha256,
                    "text": c.text,
                    "embedding": v,
                }
            )
        written_chunks += store.upsert_chunks(payload)

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
    """Ask a question about an indexed workspace and show citations."""
    ws, layout, wdir = _workspace_and_layout(path, name, local_store)
    backend = BackendOptions(embedder=embedder, llm="ollama", model="", embed_model=embed_model, ollama_host=ollama_host)
    # Persist workspace metadata (helps listing + pack portability)
    try:
        write_workspace_json(wdir, ws, backend)
    except Exception:
        pass
    store = SQLiteNumpyVectorStore(store_dir=wdir)

    embed = OllamaEmbedder(host=ollama_host, model=embed_model)
    llm = OllamaLLM(host=ollama_host, model=llm_model)

    runtime = RuntimeOptions(top_k=top_k)

    qvec = embed.embed([question])[0]
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
    """Start an interactive Q&A loop over an indexed workspace.

    This is the menu-friendly equivalent of the `wikdoc chat` command.
    Type `/exit` to leave. Type `/help` for shortcuts.
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
    """Generate Markdown documentation for a workspace using RAG + an Ollama LLM.

    Templates:
      - wiki:      multiple pages (Overview, Architecture, Key Modules)
      - readme:    single README.md
      - architecture: focused architecture doc
    """
    ws, layout, wdir = _workspace_and_layout(path, name=name, local_store=local_store)

    # Persist workspace metadata (helps listing + pack portability)
    backend = BackendOptions(embedder=embedder, llm="ollama", model="", embed_model=embed_model, ollama_host=ollama_host)
    try:
        write_workspace_json(wdir, ws, backend)
    except Exception:
        pass

    store = SQLiteNumpyVectorStore(store_dir=wdir)
    embed = _make_embedder(embedder, ollama_host, embed_model)
    llm = OllamaLLM(host=ollama_host, model=llm_model)

    out_dir = Path(out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _gather_context(queries: list[str], k: int) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for q in queries:
            qv = embed.embed([q])[0]
            hits.extend(retrieve(store, qv, k))
        # de-dupe by (file,start,end)
        seen = set()
        uniq = []
        for h in hits:
            # SearchHit may not include a `meta` dict; prefer structured fields when available.
            if hasattr(h, 'meta') and isinstance(getattr(h, 'meta'), dict):
                key = (h.meta.get('file_path'), h.meta.get('start_line'), h.meta.get('end_line'))
            else:
                key = (getattr(h, 'path', None), getattr(h, 'start_line', None), getattr(h, 'end_line', None))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(h)
        return uniq[: max(k, 12)]

    def _ctx_block(hits: list[SearchHit], max_chars: int = 24000) -> tuple[str, list[str]]:
        parts = []
        sources = []
        total = 0
        for h in hits:
            if hasattr(h, 'meta') and isinstance(getattr(h, 'meta'), dict):
                fp = h.meta.get('file_path', 'unknown')
                sl = h.meta.get('start_line', '?')
                el = h.meta.get('end_line', '?')
            else:
                fp = getattr(h, 'path', 'unknown')
                sl = getattr(h, 'start_line', '?')
                el = getattr(h, 'end_line', '?')
            src = f"{fp}:{sl}-{el}"
            chunk = h.text.strip()
            snippet = f"### {src}\n{chunk}\n"
            if total + len(snippet) > max_chars:
                break
            total += len(snippet)
            parts.append(snippet)
            sources.append(src)
        return "\n".join(parts).strip(), sources

    def _write_md(filename: str, title: str, body: str, sources: list[str]) -> Path:
        p = out_dir / filename
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        src_md = "\n".join([f"- {s}" for s in sources[:40]])
        p.write_text(
            f"# {title}\n\n"
            f"_Generated by Wikdoc on {stamp}._\n\n"
            f"{body.strip()}\n\n"
            f"---\n\n"
            f"## Sources (from your workspace)\n{src_md}\n",
            encoding="utf-8",
        )
        return p

    def _llm_doc(system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return llm.chat(messages)

    system = (
        "You are a senior software engineer writing concise, accurate project documentation. "
        "Use ONLY the provided context. If something is unknown, say so."
    )

    written: list[Path] = []

    if template == "readme":
        hits = _gather_context(
            ["project overview", "how to run", "configuration", "entry point", "architecture", "folders structure"],
            k=top_k,
        )
        ctx, sources = _ctx_block(hits)
        user = (
            "Write a README.md for this codebase. Include: Overview, Setup/Install, How to Run, "
            "Architecture, Key Modules, and Common Commands.\n\n"
            f"CONTEXT:\n{ctx}"
        )
        body = _llm_doc(system, user)
        written.append(_write_md("README.generated.md", f"{ws.name or ws.root.name} — README", body, sources))

    elif template == "architecture":
        hits = _gather_context(
            ["clean architecture", "layers", "dependency injection", "api endpoints", "data access", "services"],
            k=top_k,
        )
        ctx, sources = _ctx_block(hits)
        user = (
            "Write an architecture document for this codebase. Include: High-level diagram description (text), "
            "layers/components, data flow, key dependencies, and extension points.\n\n"
            f"CONTEXT:\n{ctx}"
        )
        body = _llm_doc(system, user)
        written.append(_write_md("Architecture.md", f"{ws.name or ws.root.name} — Architecture", body, sources))

    else:  # wiki default
        # Overview
        hits = _gather_context(["project overview", "README", "solution structure", "entry point"], k=top_k)
        ctx, sources = _ctx_block(hits)
        body = _llm_doc(system, f"Write an Overview page for this project.\n\nCONTEXT:\n{ctx}")
        written.append(_write_md("01_Overview.md", f"{ws.name or ws.root.name} — Overview", body, sources))

        # Architecture
        hits = _gather_context(["architecture", "layers", "services", "controllers", "modules"], k=top_k)
        ctx, sources = _ctx_block(hits)
        body = _llm_doc(system, f"Write an Architecture page for this project.\n\nCONTEXT:\n{ctx}")
        written.append(_write_md("02_Architecture.md", f"{ws.name or ws.root.name} — Architecture", body, sources))

        # Key Modules
        hits = _gather_context(["controllers", "services", "repositories", "application layer", "domain models"], k=top_k)
        ctx, sources = _ctx_block(hits)
        body = _llm_doc(system, f"Write a Key Modules page. Organize by folder/module and purpose.\n\nCONTEXT:\n{ctx}")
        written.append(_write_md("03_Key_Modules.md", f"{ws.name or ws.root.name} — Key Modules", body, sources))

    console.print(f"[bold green]Generated {len(written)} file(s) into[/bold green] {out_dir}")
    for p in written:
        console.print(f" - {p}")
def do_status(path: str, local_store: bool) -> None:
    ws, layout, wdir = _workspace_and_layout(path, name=None, local_store=local_store)
    store = SQLiteNumpyVectorStore(store_dir=wdir)
    stats = store.stats()
    console.print(f"Workspace: {ws.root}")
    for k, v in stats.items():
        console.print(f"- {k}: {v}")
    store.close()


def do_reset(path: str, local_store: bool) -> None:
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
    """Export current workspace index as a portable pack."""
    out_path = export_pack(path=path, local_store=local_store, out_file=out_file, name=name, include_docs=True)
    console.print(f"[green]Pack exported:[/green] {out_path}")

def do_pack_import(pack_file: str, mount_path: str, local_store: bool, name: Optional[str] = None, overwrite: bool = False) -> None:
    """Import a pack into the given mount path."""
    wdir = import_pack(pack_file=pack_file, mount_path=mount_path, local_store=local_store, name=name, overwrite=overwrite)
    console.print(f"[green]Pack imported into:[/green] {wdir}")

def do_workspaces_list() -> None:
    """List workspaces in the global store."""
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