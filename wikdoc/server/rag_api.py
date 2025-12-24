"""
Wikdoc RAG API (OpenAI-compatible) for external clients.

Goal:
- You already have a Wikdoc index (SQLite + embeddings).
- This file exposes a local HTTP API that:
  1) selects a Wikdoc workspace by id (acts as a "knowledge base")
  2) retrieves relevant chunks (RAG)
  3) calls your local Ollama model to answer using those chunks
  4) returns responses in OpenAI Chat Completions format so OpenAI-compatible clients can use it.

This is NOT OpenAI cloud. It's just a compatible local HTTP schema for local tools.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ..store.layout import StoreLayout, default_store_dir
from ..store.workspace import Workspace
from ..vectordb.sqlite_numpy import SQLiteNumpyVectorStore
from ..embeddings.ollama import OllamaEmbedder
from ..rag.retrieve import retrieve


@dataclass
class WorkspaceRef:
    """A resolved workspace directory and its metadata."""
    workspace_id: str
    wdir: Path
    root: Path
    name: Optional[str]
    scope: str  # "local" or "global"


def _load_workspace_json(wdir: Path) -> Optional[Dict[str, Any]]:
    p = wdir / "workspace.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# Historical note:
# - Older Wikdoc builds used `store.sqlite`.
# - Current SQLiteNumpyVectorStore uses `chunks.sqlite3`.
# We support both so existing user indexes keep working.
_INDEX_DB_CANDIDATES = ("chunks.sqlite3", "store.sqlite")


def _index_db_path(wdir: Path) -> Optional[Path]:
    """Return the existing index DB file path inside a workspace dir (if any)."""
    for name in _INDEX_DB_CANDIDATES:
        p = wdir / name
        if p.exists():
            return p
    return None


def _has_index(w: WorkspaceRef) -> bool:
    """True when the workspace directory has an index DB."""
    try:
        return _index_db_path(w.wdir) is not None
    except Exception:
        return False


def _dedupe_workspaces(workspaces: List[WorkspaceRef]) -> List[WorkspaceRef]:
    """De-duplicate by workspace_id.

    The same workspace_id can exist in both the global and local stores.
    We prefer the instance that actually has an index DB so clients can query it.
    """
    best: Dict[str, WorkspaceRef] = {}
    for w in workspaces:
        cur = best.get(w.workspace_id)
        if cur is None:
            best[w.workspace_id] = w
            continue

        # Prefer indexed over non-indexed
        if _has_index(w) and not _has_index(cur):
            best[w.workspace_id] = w
            continue

        # If both are indexed (or both not), keep the first one to be stable.

    return list(best.values())



def _content_to_text(content: Any) -> str:
    """Best-effort extraction of plain text from OpenAI-style message content.

    Some OpenAI-compatible clients may send:
      - str
      - list[{"type":"text","text":"..."} , ...]
      - {"type":"text","text":"..."}
    We ignore non-text parts.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # e.g. {"type":"text","text":"hello"}
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
                continue
            if isinstance(p, dict):
                if p.get("type") == "text" and isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p.get("content"), str):
                    parts.append(p["content"])
        return "\n".join([x for x in parts if x])
    # Fallback: avoid crashing on unknown types
    try:
        return str(content)
    except Exception:
        return ""


def _last_user_message_text(messages: list[dict]) -> str:
    """Return the last user message text from an OpenAI messages array."""
    for m in reversed(messages or []):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        return _content_to_text(m.get("content")).strip()
    # Some clients might use 'input' or 'prompt' only.
    return ""



def _extract_model_id(body: Dict[str, Any]) -> Optional[str]:
    """Extract model id from OpenAI-compatible payloads (best-effort)."""
    mid = body.get("model")
    if isinstance(mid, str) and mid.strip():
        return mid.strip()

    model_item = body.get("model_item")
    if isinstance(model_item, dict):
        for k in ("id", "name"):
            v = model_item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        openai_obj = model_item.get("openai")
        if isinstance(openai_obj, dict):
            v = openai_obj.get("id")
            if isinstance(v, str) and v.strip():
                return v.strip()

    return None


def _extract_workspace_root_hint(body: Dict[str, Any]) -> Optional[Path]:
    """Try to extract the workspace root from OpenAI-compatible payloads."""
    model_item = body.get("model_item")
    if isinstance(model_item, dict):
        label = model_item.get("label")
        if isinstance(label, str) and label.strip():
            try:
                return Path(label)
            except Exception:
                return None
    return None


def _locate_workspace_dir(workspace_id: str, body: Dict[str, Any]) -> Optional[Path]:
    """Locate the workspace directory that contains store.sqlite.

    We try (in this order):
      1) Local store computed from the workspace label (some UIs send it)
      2) Local store computed from the current working directory
      3) Global store (~/.wikdoc)

    Returns the first workspace dir found. If store.sqlite isn't found but a
    workspace.json exists, returns that dir so the caller can provide a better
    error message.
    """
    candidates: list[Path] = []

    # 1) local store: workspace label/root from the UI
    root_hint = _extract_workspace_root_hint(body)
    if root_hint is not None:
        try:
            candidates.append(default_store_dir(local_store=True, workspace_root=root_hint) / "workspaces" / workspace_id)
        except Exception:
            pass

    # 2) local store: current working directory
    try:
        candidates.append(default_store_dir(local_store=True, workspace_root=Path.cwd()) / "workspaces" / workspace_id)
    except Exception:
        pass

    # 3) global store
    candidates.append(default_store_dir(local_store=False, workspace_root=Path.cwd()) / "workspaces" / workspace_id)

    for wdir in candidates:
        if (wdir / "store.sqlite").exists():
            return wdir

    for wdir in candidates:
        if (wdir / "workspace.json").exists():
            return wdir

    return None


def _iter_workspace_dirs(base_dir: Path) -> Iterable[Path]:
    ws_root = base_dir / "workspaces"
    if not ws_root.exists():
        return []
    return [p for p in ws_root.iterdir() if p.is_dir()]


def list_workspaces() -> List[WorkspaceRef]:
    """List workspaces from both global (~/.wikdoc) and local (<cwd>/.wikdoc)."""
    out: List[WorkspaceRef] = []

    # global
    # default_store_dir signature is (local_store, workspace_root)
    # workspace_root is ignored when local_store=False, but required by the signature.
    global_dir = default_store_dir(local_store=False, workspace_root=Path.cwd())
    for wdir in _iter_workspace_dirs(global_dir):
        meta = _load_workspace_json(wdir) or {}
        out.append(
            WorkspaceRef(
                workspace_id=wdir.name,
                wdir=wdir,
                root=Path(meta.get("root") or meta.get("workspace_root") or ""),
                name=meta.get("name"),
                scope="global",
            )
        )

    # local (cwd)
    local_dir = default_store_dir(local_store=True, workspace_root=Path.cwd())
    for wdir in _iter_workspace_dirs(local_dir):
        meta = _load_workspace_json(wdir) or {}
        out.append(
            WorkspaceRef(
                workspace_id=wdir.name,
                wdir=wdir,
                root=Path(meta.get("root") or meta.get("workspace_root") or ""),
                name=meta.get("name"),
                scope="local",
            )
        )
    # De-dupe by id.
    # Prefer the entry that actually has an index (store.sqlite). If tied, prefer local.
    seen = set()
    uniq: List[WorkspaceRef] = []
    for w in sorted(out, key=lambda x: (0 if _has_index(x) else 1, 0 if x.scope == "local" else 1)):
        if w.workspace_id in seen:
            continue
        seen.add(w.workspace_id)
        uniq.append(w)
    return uniq


def resolve_workspace(workspace_id: str) -> WorkspaceRef:
    """Resolve a workspace id to its directory (local preferred)."""
    for w in list_workspaces():
        if w.workspace_id == workspace_id:
            return w
    raise HTTPException(status_code=404, detail=f"Unknown workspace id: {workspace_id}")


def _ensure_index_exists(w: WorkspaceRef) -> Path:
    db = _index_db_path(w.wdir)
    if db is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Index not found for this workspace (no index DB present). "
                "Run Wikdoc → Index workspace first (same store scope)."
            ),
        )
    # Return the workspace directory (store root). The vector store expects a directory.
    return w.wdir


def _open_store(w: WorkspaceRef) -> SQLiteNumpyVectorStore:
    _ensure_index_exists(w)
    return SQLiteNumpyVectorStore(w.wdir)


def _open_embedder(ollama_host: str, embed_model: str) -> OllamaEmbedder:
    return OllamaEmbedder(host=ollama_host, model=embed_model)


def _build_context(hits: List[Dict[str, Any]], max_chars: int = 24000) -> Tuple[str, List[str]]:
    """
    Build a context string + a short list of sources.

    Args:
        hits: list of SearchHit-like dicts (path/start_line/end_line/text)
        max_chars: cap prompt size

    Returns:
        (context, sources)
    """
    parts: List[str] = []
    sources: List[str] = []
    total = 0
    for h in hits:
        src = f"{h.get('path')}:{h.get('start_line')}-{h.get('end_line')}"
        if src not in sources:
            sources.append(src)
        snippet = h.get("text") or ""
        block = f"\n---\nSOURCE: {src}\n{snippet}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    ctx = "".join(parts).strip()
    return ctx, sources[:20]


def _ollama_chat(
    *,
    host: str,
    model: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    timeout: int = 600,
) -> Any:
    url = f"{host.rstrip('/')}/api/chat"
    payload = {"model": model, "messages": messages, "stream": stream}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r


def create_app(
    *,
    ollama_host: str = "http://localhost:11434",
    llm_model: str = "qwen2.5-coder:7b",
    embed_model: str = "nomic-embed-text",
    top_k: int = 8,
    # Store selection (matches CLI flags)
    local_store: bool = False,
    workspace_path: Optional[str] = None,
    store_dir: Optional[str] = None,
) -> FastAPI:
    app = FastAPI(title="Wikdoc RAG API", version="0.3.0")


    # CORS: browser clients may preflight with OPTIONS; allow local dev by default.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------------
    # Store / workspace resolution
    # ---------------------------------------------------------------------
    workspace_root = Path(workspace_path).expanduser().resolve() if workspace_path else Path.cwd()

    # Primary store used by the API (what /v1/models exposes first)
    if store_dir:
        primary_base_dir = Path(store_dir).expanduser().resolve()
    else:
        primary_base_dir = default_store_dir(local_store=local_store, workspace_root=workspace_root)
    primary_layout = StoreLayout(primary_base_dir)

    # Fallback stores (helpful when a client runs from a different CWD)
    global_layout = StoreLayout(default_store_dir(local_store=False, workspace_root=workspace_root))
    local_layout = StoreLayout(default_store_dir(local_store=True, workspace_root=workspace_root))



    def list_workspaces_in_dir(base_dir: Path, scope: str) -> List[WorkspaceRef]:
        out: List[WorkspaceRef] = []
        if not base_dir.exists():
            return out

        for wdir in sorted(base_dir.iterdir()):
            if not wdir.is_dir():
                continue

            meta = _load_workspace_json(wdir)
            workspace_id = meta.get("workspace_id") or meta.get("id") or wdir.name
            name = meta.get("name") or wdir.name

            out.append(
                WorkspaceRef(
                    workspace_id=str(workspace_id),
                    wdir=wdir,
                    root=wdir,
                    name=str(name) if name is not None else None,
                    scope=scope,
                )
            )

        return out

    def list_workspaces() -> List[WorkspaceRef]:
        """List workspaces for the configured store mode.

        - If --store-dir is provided, list ONLY that store.
        - If --local-store is set, list local store (for --path) and also global as fallback.
        - Otherwise, list global store and also local (for cwd/path) as fallback.
        """
        candidates: List[Tuple[str, StoreLayout]] = []
        if store_dir:
            candidates = [("store", primary_layout)]
        elif local_store:
            candidates = [("local", local_layout), ("global", global_layout)]
        else:
            candidates = [("global", global_layout), ("local", local_layout)]

        seen: set[str] = set()
        out: List[WorkspaceRef] = []
        for scope, layout in candidates:
            for ws in list_workspaces_in_dir(layout.base_dir, scope=scope):
                if ws.workspace_id in seen:
                    continue
                # Only expose workspaces that actually have an index.
                if not _has_index(ws):
                    continue
                seen.add(ws.workspace_id)
                out.append(ws)
        return out

    def resolve_workspace(workspace_id: str) -> WorkspaceRef:
        for w in list_workspaces():
            if w.workspace_id == workspace_id:
                return w
        raise KeyError(f"workspace not found: {workspace_id}")

    def _locate_workspace_dir(workspace_id: str, body: Dict[str, Any]) -> Optional[Path]:
        """Locate the store/workspace folder that contains store.sqlite for a workspace id."""
        root_hint = None
        model_item = body.get("model_item") or {}
        if isinstance(model_item, dict):
            root_hint = model_item.get("label") or model_item.get("name")
            # Some clients include a full path in label/name
            if isinstance(root_hint, str) and ("\\" in root_hint or "/" in root_hint):
                try:
                    root_hint = str(Path(root_hint).expanduser().resolve())
                except Exception:
                    pass

        candidates: List[Path] = []

        # If we have a root hint, try that local store first.
        if isinstance(root_hint, str) and root_hint:
            try:
                hinted_root = Path(root_hint).expanduser().resolve()
                hinted_local = default_store_dir(local_store=True, workspace_root=hinted_root)
                candidates.append(StoreLayout(hinted_local).workspace_dir(workspace_id))
            except Exception:
                pass

        # Primary store
        candidates.append(primary_layout.workspace_dir(workspace_id))

        # Fallback stores
        candidates.append(local_layout.workspace_dir(workspace_id))
        candidates.append(global_layout.workspace_dir(workspace_id))

        # Choose the first one that contains a SQLite index
        for c in candidates:
            if (c / "store.sqlite").exists():
                return c
        return None

    @app.get("/health")
    def health():
        return {"ok": True, "time": int(time.time())}

    # Some clients probe the OpenAI base prefix directly (GET /v1) to
    # validate a connection. OpenAI's API doesn't define a root payload,
    # but returning a small JSON object makes those clients happy.
    @app.get("/v1")
    def v1_root():
        return {
            "ok": True,
            "message": "Wikdoc OpenAI-compatible endpoint",
        }

    @app.get("/v1/models")
    def v1_models():
        # One "model" per workspace so OpenAI-compatible clients can select a KB.
        models = []
        for w in list_workspaces():
            # Only expose workspaces that actually have an index.
            # This avoids clients selecting a workspace that exists in metadata
            # but has never been indexed (store.sqlite missing).
            if not _has_index(w):
                continue
            label = w.name or str(w.root) or w.workspace_id
            models.append(
                {
                    "id": f"wikdoc:{w.workspace_id}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "wikdoc",
                    "label": label,
                    "scope": w.scope,
                }
            )
        return {"object": "list", "data": models}

    def _workspace_from_model_id(model_id: str) -> WorkspaceRef:
        """Resolve a model id sent by an OpenAI-compatible client.

        Preferred: wikdoc:<workspace_id>
        Fallbacks:
          - raw workspace id
          - label match (some clients may send the 'label' instead of the id in some flows)
        """
        if model_id.startswith("wikdoc:"):
            wsid = model_id.split(":", 1)[1]
            return resolve_workspace(wsid)

        # allow passing raw workspace id too
        try:
            return resolve_workspace(model_id)
        except HTTPException:
            pass

        # label match fallback
        for w in list_workspaces():
            if w.label == model_id or (w.name and w.name == model_id):
                return w
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(req: Request):
        body = await req.json()
        model_id = _extract_model_id(body) or ""
        stream = bool(body.get("stream", False))
        messages = body.get("messages") or []

        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model'. Use one from GET /v1/models")
        if not messages:
            raise HTTPException(status_code=400, detail="Missing 'messages'.")

        # Resolve workspace directory robustly (clients may run from a different CWD)
        workspace_id = model_id.split(":", 1)[1] if model_id.startswith("wikdoc:") else model_id
        wdir = _locate_workspace_dir(workspace_id, body)
        if wdir is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Workspace index not found for this model. "
                    "If you indexed with --local-store, run: wikdoc serve --local-store --path <workspace_root>."
                ),
            )

        # Load workspace metadata if present
        try:
            ws_meta = Workspace.from_dir(wdir)
        except Exception:
            root_hint = _extract_workspace_root_hint(body) or Path.cwd()
            ws_meta = Workspace(root=root_hint, name=None)

        scope = "local" if (root_hint := _extract_workspace_root_hint(body)) and str(wdir).startswith(str(root_hint)) else "global"
        w = WorkspaceRef(
            workspace_id=workspace_id,
            root=ws_meta.root,
            name=ws_meta.name,
            scope=scope,
            wdir=wdir,
        )

        store = _open_store(w)
        embed = _open_embedder(ollama_host, embed_model)

        # Use the last user message as the retrieval query
        query = _last_user_message_text(messages)
        if not query:
            # Some clients may send a single-string prompt/input instead of messages.
            if isinstance(body.get("prompt"), str):
                query = body["prompt"].strip()
            elif isinstance(body.get("input"), str):
                query = body["input"].strip()
        if not query:
            raise HTTPException(status_code=400, detail="Empty user query.")

        # RAG retrieve
        hits = retrieve(store=store, embedder=embed, query=query, k=top_k)
        # hits are SearchHit objects; convert to dict
        hit_dicts = [h.__dict__ if hasattr(h, "__dict__") else dict(h) for h in hits]
        context, sources = _build_context(hit_dicts)

        sys = (
            "You are a helpful assistant answering questions about a codebase.\n"
            "Use ONLY the context below when possible. If something is not in the context, say you can't find it.\n"
            "When relevant, mention file paths and line ranges.\n\n"
            f"CONTEXT:\n{context}\n"
        )

        # Build messages for Ollama
        ollama_messages = [{"role": "system", "content": sys}]
        # Keep a small window of prior user/assistant messages (avoid huge prompts)
        for m in messages[-12:]:
            role = m.get("role")
            if role in ("user", "assistant"):
                ollama_messages.append({"role": role, "content": _content_to_text(m.get("content"))})

        if not stream:
            r = _ollama_chat(host=ollama_host, model=llm_model, messages=ollama_messages, stream=False)
            data = r.json()
            text = (data.get("message") or {}).get("content") or ""
            if sources:
                text += "\n\n---\nTop sources:\n" + "\n".join([f"- {s}" for s in sources])
            resp = {
                "id": f"chatcmpl_{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            }
            return JSONResponse(resp)

        # Streaming: translate Ollama NDJSON stream into OpenAI SSE stream
        def sse() -> Iterable[bytes]:
            # first delta chunk (empty)
            yield b"data: " + json.dumps(
                {
                    "id": f"chatcmpl_{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            ).encode("utf-8") + b"\n\n"

            r = _ollama_chat(host=ollama_host, model=llm_model, messages=ollama_messages, stream=True)
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    j = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                msg = (j.get("message") or {}).get("content")
                if msg:
                    yield b"data: " + json.dumps(
                        {
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {"content": msg}, "finish_reason": None}],
                        }
                    ).encode("utf-8") + b"\n\n"
                if j.get("done"):
                    break

            # append sources at end (one chunk)
            if sources:
                tail = "\n\n---\nTop sources:\n" + "\n".join([f"- {s}" for s in sources])
                yield b"data: " + json.dumps(
                    {
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}],
                    }
                ).encode("utf-8") + b"\n\n"

            yield b"data: " + json.dumps(
                {
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ).encode("utf-8") + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app
