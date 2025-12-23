"""SQLite + NumPy vector store (local-first, zero extra services).

Storage:
  - Metadata + chunk text in SQLite
  - Embeddings as `.npy` files on disk (one per chunk)

Retrieval:
  - Loads vectors and computes cosine similarity in NumPy (MVP).
  - For very large workspaces, consider an ANN backend in the future.
"""

from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .base import SearchHit, VectorStore


def _ensure_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            language TEXT NOT NULL,
            symbol TEXT,
            file_sha256 TEXT NOT NULL,
            text TEXT NOT NULL,
            emb_path TEXT NOT NULL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);")
    conn.commit()


class SQLiteNumpyVectorStore(VectorStore):
    """Vector store implementation backed by SQLite + `.npy` files."""

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = store_dir
        self.db_path = store_dir / "chunks.sqlite3"
        self.emb_dir = store_dir / "embeddings"
        self.emb_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        _ensure_db(self._conn)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def upsert_chunks(self, chunks: List[dict]) -> int:
        """Upsert chunks into the store.

        Args:
            chunks: List of chunk dicts containing:
                - file_path, start_line, end_line, language, symbol
                - file_sha256, text, embedding (list[float])

        Returns:
            Number of chunks written.
        """
        cur = self._conn.cursor()
        written = 0
        for ch in chunks:
            chunk_id = ch.get("chunk_id") or uuid.uuid4().hex
            emb = np.asarray(ch["embedding"], dtype=np.float32)

            emb_path = self.emb_dir / f"{chunk_id}.npy"
            np.save(str(emb_path), emb)

            cur.execute(
                """
                INSERT INTO chunks(chunk_id, file_path, start_line, end_line, language, symbol, file_sha256, text, emb_path)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    file_path=excluded.file_path,
                    start_line=excluded.start_line,
                    end_line=excluded.end_line,
                    language=excluded.language,
                    symbol=excluded.symbol,
                    file_sha256=excluded.file_sha256,
                    text=excluded.text,
                    emb_path=excluded.emb_path
                """,
                (
                    chunk_id,
                    ch["file_path"],
                    int(ch["start_line"]),
                    int(ch["end_line"]),
                    ch["language"],
                    ch.get("symbol"),
                    ch["file_sha256"],
                    ch["text"],
                    str(emb_path),
                ),
            )
            written += 1
        self._conn.commit()
        return written

    def _iter_all_embeddings(self):
        cur = self._conn.cursor()
        for row in cur.execute(
            "SELECT chunk_id, file_path, start_line, end_line, language, symbol, text, emb_path FROM chunks"
        ):
            yield row

    def search(self, query_vec: Sequence[float], top_k: int) -> List[SearchHit]:
        q = np.asarray(query_vec, dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-12)

        hits: List[SearchHit] = []
        for (chunk_id, file_path, start_line, end_line, language, symbol, text, emb_path) in self._iter_all_embeddings():
            try:
                v = np.load(emb_path).astype(np.float32)
                vn = v / (np.linalg.norm(v) + 1e-12)
                score = float(np.dot(qn, vn))
                hits.append(
                    SearchHit(
                        chunk_id=chunk_id,
                        score=score,
                        path=file_path,
                        start_line=int(start_line),
                        end_line=int(end_line),
                        text=text,
                        language=language,
                        symbol=symbol,
                    )
                )
            except Exception:
                continue

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: max(1, int(top_k))]

    def stats(self) -> dict:
        cur = self._conn.cursor()
        n = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"chunks": int(n), "db_path": str(self.db_path), "emb_dir": str(self.emb_dir)}

    def reset(self) -> None:
        cur = self._conn.cursor()
        cur.execute("DROP TABLE IF EXISTS chunks")
        self._conn.commit()
        _ensure_db(self._conn)
        for p in self.emb_dir.glob("*.npy"):
            try:
                p.unlink()
            except Exception:
                pass
