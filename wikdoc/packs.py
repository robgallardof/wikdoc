"""Pack (export/import) utilities.

A pack is a portable zip bundle that contains:
  - the vector store (SQLite) + metadata
  - the manifest.json (file hashes)
  - generated docs (optional)
  - a workspace.json describing how it was built (embed model, etc.)

This is NOT "model training". It's a RAG index snapshot that can be shared.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

from .config import BackendOptions, StoreLayout, Workspace, default_store_dir


PACK_EXT = ".wikdocpack.zip"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _workspace_and_layout(
    path: str,
    name: Optional[str],
    local_store: bool,
    *,
    ensure: bool = True,
) -> Tuple[Workspace, StoreLayout, Path]:
    """Resolve workspace + store layout + workspace dir.

    Notes:
        - When *exporting*, we should NOT create an empty workspace dir; we should
          look for an existing index on disk. Use ensure=False for that case.
    """
    ws = Workspace.from_path(path, name=name)
    store_root = default_store_dir(local_store=local_store, workspace_root=ws.root)
    layout = StoreLayout(base_dir=store_root)
    wdir = layout.ensure(ws) if ensure else layout.workspace_dir(ws)
    return ws, layout, wdir


def write_workspace_json(wdir: Path, ws: Workspace, backend: BackendOptions) -> None:
    """Write a small JSON descriptor for the workspace index."""
    data = {
        "name": ws.name,
        "root": str(ws.root),
        "workspace_id": ws.id,
        "created_at": _now_iso(),
        "backend": asdict(backend),
    }
    (wdir / "workspace.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def export_pack(
    *,
    path: str,
    local_store: bool,
    out_file: str,
    name: Optional[str] = None,
    include_docs: bool = True,
) -> Path:
    """Export a workspace index as a portable pack (zip)."""
    ws, layout, wdir = _workspace_and_layout(path, name=name, local_store=local_store, ensure=False)

    out_path = Path(out_file)
    if out_path.suffix.lower() != ".zip":
        out_path = out_path.with_suffix(out_path.suffix + ".zip") if out_path.suffix else out_path.with_suffix(".zip")

    if not out_path.name.endswith(PACK_EXT):
        # keep user's name, but make it recognizable
        out_path = out_path.with_name(out_path.stem + PACK_EXT)

    meta = {
        "format": "wikdoc-pack-v1",
        "created_at": _now_iso(),
        "workspace_name": ws.name,
        "workspace_id": ws.id,
        "source_root_hint": str(ws.root),
    }

    # Ensure there's at least something to export.
    #
    # Users can index either in "local store" (<workspace>/.wikdoc) or in the
    # global store (~/.wikdoc). If the caller picks the wrong mode, exporting
    # should still "just work" by auto-detecting where the index lives.
    #
    # Historical note:
    # - Older Wikdoc prototypes used `store.sqlite`.
    # - Current SQLiteNumpyVectorStore uses `chunks.sqlite3`.
    store_files = ("chunks.sqlite3", "store.sqlite")

    # Primary (requested) location
    wdir_primary = wdir
    store_root_alt = default_store_dir(local_store=not local_store, workspace_root=ws.root)
    layout_alt = StoreLayout(base_dir=store_root_alt)
    wdir_alt = layout_alt.workspace_dir(ws)

    candidates = [
        ("primary", wdir_primary),
        ("alternate", wdir_alt),
    ]

    chosen_dir: Optional[Path] = None
    chosen_label: Optional[str] = None
    chosen_db: Optional[str] = None
    for label, d in candidates:
        for sf in store_files:
            if (d / sf).exists():
                chosen_dir = d
                chosen_label = label
                chosen_db = sf
                break
        if chosen_dir is not None:
            break

    if chosen_dir is None:
        checked = []
        for _, d in candidates:
            checked.extend([str(d / sf) for sf in store_files])
        raise FileNotFoundError(
            "No index found (index DB missing). "
            "Run 'Index workspace' first, or switch Local/Global store.\n"
            f"Checked: {checked}"
        )

    # If we detected the index in the alternate location, use that directory.
    wdir = chosen_dir
    meta["store_scope"] = "local" if local_store else "global"
    if chosen_label == "alternate":
        # We requested one scope but found the other.
        meta["store_scope"] = "global" if local_store else "local"

    if chosen_db:
        meta["index_db"] = chosen_db

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.json", json.dumps(meta, indent=2))

        for p in wdir.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(wdir)
            if not include_docs and str(rel).startswith("docs/"):
                continue
            # Keep cache / temp out of packs if any
            if "__pycache__" in p.parts:
                continue
            z.write(str(p), f"workspace/{rel.as_posix()}")

    return out_path


def import_pack(
    *,
    pack_file: str,
    mount_path: str,
    local_store: bool,
    name: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Import a pack into a new (or existing) workspace storage directory.

    Args:
        pack_file: Path to the .wikdocpack.zip
        mount_path: Folder on this machine that represents the workspace root.
        local_store: If True, import into <mount_path>/.wikdoc, else into ~/.wikdoc
        name: Optional friendly name for this workspace
        overwrite: If True, overwrite existing workspace dir
    """
    pack_path = Path(pack_file)
    if not pack_path.exists():
        raise FileNotFoundError(pack_file)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with zipfile.ZipFile(pack_path, "r") as z:
            z.extractall(tmp)

        meta_path = tmp / "meta.json"
        if not meta_path.exists():
            raise ValueError("Invalid pack: meta.json missing.")

        ws, layout, wdir = _workspace_and_layout(mount_path, name=name, local_store=local_store)

        src_w = tmp / "workspace"
        if not src_w.exists():
            raise ValueError("Invalid pack: workspace/ directory missing.")

        # If destination already has data, protect the user
        if wdir.exists() and any(wdir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Workspace store already exists: {wdir}. Use overwrite=True to replace it."
                )
            shutil.rmtree(wdir)

        # Recreate structure
        (wdir / "embeddings").mkdir(parents=True, exist_ok=True)
        (wdir / "docs").mkdir(parents=True, exist_ok=True)

        # Copy files
        for p in src_w.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(src_w)
            dest = wdir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)

    return wdir


def list_global_workspaces() -> list[dict]:
    """List workspaces available in the *global* store (~/.wikdoc)."""
    base_dir = default_store_dir(local_store=False, workspace_root=Path.home())
    layout = StoreLayout(base_dir=base_dir)

    out: list[dict] = []
    workspaces_dir = base_dir / "workspaces"
    if not workspaces_dir.exists():
        return out

    for wdir in sorted(workspaces_dir.iterdir()):
        if not wdir.is_dir():
            continue
        info_path = wdir / "workspace.json"
        item = {"workspace_dir": str(wdir)}
        if info_path.exists():
            try:
                item.update(json.loads(info_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        # fallback: look for manifest
        mp = wdir / "manifest.json"
        if mp.exists() and "workspace_id" not in item:
            item["workspace_id"] = wdir.name
        out.append(item)

    return out