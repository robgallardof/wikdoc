# wikdoc/server/rag_api.py
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

# Historical note:
# - Older Wikdoc builds used `store.sqlite`.
# - Current SQLiteNumpyVectorStore uses `chunks.sqlite3`.
# We support both so existing user indexes keep working.
_INDEX_DB_CANDIDATES = ("chunks.sqlite3", "store.sqlite")


@dataclass
class WorkspaceRef:
    """
    A resolved workspace directory and its metadata.

    Attributes:
        workspace_id: Stable workspace id (string).
        wdir: Store directory for this workspace (contains sqlite db, manifest, workspace.json).
        root: Workspace root path (source code folder).
        name: Optional friendly name.
        scope: "local" or "global" (store scope where we found it).
    """

    workspace_id: str
    wdir: Path
    root: Path
    name: Optional[str]
    scope: str  # "local" or "global"


def _load_workspace_json(wdir: Path) -> Optional[Dict[str, Any]]:
    """
    Load workspace.json from a store directory.

    Args:
        wdir: Workspace store directory.

    Returns:
        Parsed dict or None if missing/unreadable.
    """
    p = wdir / "workspace.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _index_db_path(wdir: Path) -> Optional[Path]:
    """
    Return the existing index DB file path inside a workspace dir (if any).

    Args:
        wdir: Workspace store directory.

    Returns:
        Path to sqlite db if present, else None.
    """
    for name in _INDEX_DB_CANDIDATES:
        p = wdir / name
        if p.exists():
            return p
    return None


def _has_index(w: WorkspaceRef) -> bool:
    """
    True when the workspace directory has an index DB.

    Args:
        w: WorkspaceRef.

    Returns:
        True if the index sqlite file exists.
    """
    try:
        return _index_db_path(w.wdir) is not None
    except Exception:
        return False


def _content_to_text(content: Any) -> str:
    """
    Best-effort extraction of plain text from OpenAI-style message content.

    Some OpenAI-compatible clients may send:
      - str
      - list[{"type":"text","text":"..."} , ...]
      - {"type":"text","text":"..."}

    We ignore non-text parts.

    Args:
        content: Content payload.

    Returns:
        Text content (may be empty).
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
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
    try:
        return str(content)
    except Exception:
        return ""


def _last_user_message_text(messages: list[dict]) -> str:
    """
    Return the last user message text from an OpenAI messages array.

    Args:
        messages: OpenAI messages array.

    Returns:
        Last user message text (trimmed) or empty string.
    """
    for m in reversed(messages or []):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        return _content_to_text(m.get("content")).strip()
    return ""


def _extract_model_id(body: Dict[str, Any]) -> Optional[str]:
    """
    Extract model id from OpenAI-compatible payloads (best-effort).

    Args:
        body: Request JSON.

    Returns:
        Model id string or None.
    """
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
    """
    Try to extract the workspace root from OpenAI-compatible payloads.

    Some clients pass a workspace path inside model_item.label.

    Args:
        body: Request JSON.

    Returns:
        Path hint or None.
    """
    model_item = body.get("model_item")
    if isinstance(model_item, dict):
        label = model_item.get("label")
        if isinstance(label, str) and label.strip():
            try:
                return Path(label)
            except Exception:
                return None
    return None


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
    timeout: int = 600000,
) -> requests.Response:
    """
    Call Ollama chat endpoint.

    Args:
        host: Ollama base URL.
        model: Model name.
        messages: Chat messages array.
        stream: Whether to request NDJSON streaming.
        timeout: HTTP timeout seconds.

    Returns:
        Response object (caller handles .json() or .iter_lines()).

    Raises:
        requests.RequestException: On HTTP/network issues.
    """
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
    """
    Create the FastAPI app that exposes an OpenAI-compatible endpoint.

    Args:
        ollama_host: Ollama base URL.
        llm_model: LLM model for chat.
        embed_model: Embedding model.
        top_k: Retrieval size.
        local_store: Serve local store (<workspace>/.wikdoc) when True; otherwise global (~/.wikdoc).
        workspace_path: Workspace root path (required when local_store=True in CLI).
        store_dir: Override store base directory (advanced).

    Returns:
        FastAPI app.
    """
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

    def _iter_workspace_dirs(base_dir: Path) -> Iterable[Path]:
        """
        Iterate workspace store directories under <base_dir>/workspaces.

        Args:
            base_dir: Store base directory.

        Returns:
            Iterable of workspace dirs (Path).
        """
        ws_root = base_dir / "workspaces"
        if not ws_root.exists():
            return []
        return [p for p in ws_root.iterdir() if p.is_dir()]

    def _workspace_ref_from_dir(wdir: Path, scope: str) -> WorkspaceRef:
        """
        Build WorkspaceRef from a workspace store dir.

        Args:
            wdir: Store workspace directory.
            scope: "local" or "global" or "store".

        Returns:
            WorkspaceRef
        """
        meta = _load_workspace_json(wdir) or {}
        root = meta.get("root") or meta.get("workspace_root") or ""
        name = meta.get("name")
        return WorkspaceRef(
            workspace_id=str(meta.get("workspace_id") or meta.get("id") or wdir.name),
            wdir=wdir,
            root=Path(root) if isinstance(root, str) and root else Path.cwd(),
            name=str(name) if name is not None else None,
            scope=scope,
        )

    def list_workspaces() -> List[WorkspaceRef]:
        """
        List workspaces for the configured store mode.

        - If --store-dir is provided, list ONLY that store.
        - If --local-store is set, list local store (for --path) and also global as fallback.
        - Otherwise, list global store and also local (for cwd/path) as fallback.

        Returns:
            List of WorkspaceRef objects (indexed only).
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
            for wdir in _iter_workspace_dirs(layout.base_dir):
                ws = _workspace_ref_from_dir(wdir, scope=scope)
                if ws.workspace_id in seen:
                    continue
                if not _has_index(ws):
                    continue
                seen.add(ws.workspace_id)
                out.append(ws)
        return out

    def resolve_workspace(workspace_id: str) -> WorkspaceRef:
        """
        Resolve a workspace id to its directory (store order per list_workspaces).

        Args:
            workspace_id: Workspace id string.

        Returns:
            WorkspaceRef

        Raises:
            KeyError: If not found.
        """
        for w in list_workspaces():
            if w.workspace_id == workspace_id:
                return w
        raise KeyError(f"workspace not found: {workspace_id}")

    def _locate_workspace_dir(workspace_id: str, body: Dict[str, Any]) -> Optional[Path]:
        """
        Locate the store/workspace folder that contains an index DB for a workspace id.

        This is robust to clients running from a different CWD and may try:
          - hinted local store from model_item.label/name (if it's a path)
          - primary store
          - local fallback
          - global fallback

        Args:
            workspace_id: Workspace id (raw, not prefixed).
            body: Request JSON (may contain hints).

        Returns:
            Workspace dir that contains chunks.sqlite3/store.sqlite, or None.
        """
        root_hint = None
        model_item = body.get("model_item") or {}
        if isinstance(model_item, dict):
            root_hint = model_item.get("label") or model_item.get("name")
            if isinstance(root_hint, str) and ("\\" in root_hint or "/" in root_hint):
                try:
                    root_hint = str(Path(root_hint).expanduser().resolve())
                except Exception:
                    pass

        candidates: List[Path] = []

        if isinstance(root_hint, str) and root_hint:
            try:
                hinted_root = Path(root_hint).expanduser().resolve()
                hinted_local = default_store_dir(local_store=True, workspace_root=hinted_root)
                candidates.append(StoreLayout(hinted_local).workspace_dir(Workspace(root=hinted_root, name=None)))
            except Exception:
                pass

        # Store dirs (workspace_id points to store folder name, not workspace root hash here)
        candidates.append(primary_layout.base_dir / "workspaces" / workspace_id)
        candidates.append(local_layout.base_dir / "workspaces" / workspace_id)
        candidates.append(global_layout.base_dir / "workspaces" / workspace_id)

        for c in candidates:
            if _index_db_path(c) is not None:
                return c
        return None

    def _workspace_from_model_id(model_id: str) -> WorkspaceRef:
        """
        Resolve a model id sent by an OpenAI-compatible client.

        Preferred: wikdoc:<workspace_id>
        Fallbacks:
          - raw workspace id

        Args:
            model_id: Model id string.

        Returns:
            WorkspaceRef

        Raises:
            HTTPException: On unknown model.
        """
        if model_id.startswith("wikdoc:"):
            wsid = model_id.split(":", 1)[1]
            try:
                return resolve_workspace(wsid)
            except KeyError:
                raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")

        try:
            return resolve_workspace(model_id)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")

    def _open_store(wdir: Path) -> SQLiteNumpyVectorStore:
        """
        Open vector store for a workspace dir.

        Args:
            wdir: Workspace store directory.

        Returns:
            SQLiteNumpyVectorStore

        Raises:
            HTTPException: If index DB is missing.
        """
        if _index_db_path(wdir) is None:
            raise HTTPException(status_code=400, detail="Index not found for this workspace (no index DB present).")
        return SQLiteNumpyVectorStore(store_dir=wdir)

    def _open_embedder() -> OllamaEmbedder:
        """
        Open embedding backend used by the server.

        Returns:
            OllamaEmbedder
        """
        return OllamaEmbedder(host=ollama_host, model=embed_model)

    @app.get("/health")
    def health():
        return {"ok": True, "time": int(time.time())}

    @app.get("/v1")
    def v1_root():
        return {"ok": True, "message": "Wikdoc OpenAI-compatible endpoint"}

    @app.get("/v1/models")
    def v1_models():
        """
        List available "models" (one per indexed workspace).

        OpenAI-compatible clients use this to select a KB.

        Returns:
            OpenAI list payload.
        """
        models = []
        now = int(time.time())
        for w in list_workspaces():
            label = w.name or str(w.root) or w.workspace_id
            models.append(
                {
                    "id": f"wikdoc:{w.workspace_id}",
                    "object": "model",
                    "created": now,
                    "owned_by": "wikdoc",
                    "label": label,
                    "scope": w.scope,
                }
            )
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(req: Request):
        """
        OpenAI-compatible Chat Completions endpoint.

        Request:
          - model: "wikdoc:<workspace_id>"
          - messages: OpenAI messages array
          - stream: bool (optional)

        Response:
          - OpenAI chat.completion or chat.completion.chunk (SSE)

        Raises:
          - 400 for invalid payload
          - 404/400 for missing workspace index
        """
        body = await req.json()
        model_id = _extract_model_id(body) or ""
        stream = bool(body.get("stream", False))
        messages = body.get("messages") or []

        if not model_id:
            raise HTTPException(status_code=400, detail="Missing 'model'. Use one from GET /v1/models")
        if not messages and not isinstance(body.get("prompt"), str) and not isinstance(body.get("input"), str):
            raise HTTPException(status_code=400, detail="Missing 'messages' (or prompt/input).")

        # Workspace selection
        ws = _workspace_from_model_id(model_id)
        workspace_id = ws.workspace_id

        # Resolve workspace directory robustly (clients may run from a different CWD)
        wdir = _locate_workspace_dir(workspace_id, body) or ws.wdir
        if _index_db_path(wdir) is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Workspace index not found for this model. "
                    "If you indexed with --local-store, run: wikdoc serve --local-store --path <workspace_root>."
                ),
            )

        store = _open_store(wdir)
        embed = _open_embedder()

        # Retrieval query = last user message (fallback to prompt/input)
        query = _last_user_message_text(messages)
        if not query:
            if isinstance(body.get("prompt"), str):
                query = body["prompt"].strip()
            elif isinstance(body.get("input"), str):
                query = body["input"].strip()
        if not query:
            raise HTTPException(status_code=400, detail="Empty user query.")

        # RAG retrieve (vector search)
        qvec = embed.embed([query])[0]
        if not qvec:
            raise HTTPException(status_code=500, detail="Query embedding is empty. Check Ollama embedding response/model.")
        hits = retrieve(store=store, query_vec=qvec, top_k=top_k)

        hit_dicts = [
            h.__dict__ if hasattr(h, "__dict__") else dict(h)  # type: ignore[arg-type]
            for h in hits
        ]
        context, sources = _build_context(hit_dicts)

        sys = (
            "You are a helpful assistant answering questions about a codebase.\n"
            "Use ONLY the context below when possible. If something is not in the context, say you can't find it.\n"
            "When relevant, mention file paths and line ranges.\n\n"
            f"CONTEXT:\n{context}\n"
        )

        # Build messages for Ollama (small rolling window)
        ollama_messages: List[Dict[str, str]] = [{"role": "system", "content": sys}]
        for m in (messages or [])[-12:]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role in ("user", "assistant"):
                ollama_messages.append({"role": role, "content": _content_to_text(m.get("content"))})

        if not stream:
            r = _ollama_chat(host=ollama_host, model=llm_model, messages=ollama_messages, stream=False)
            data = r.json() or {}
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
