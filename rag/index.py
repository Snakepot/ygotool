"""向量库（★ 学习点）。

实现：sqlite 存 chunk 原文与向量（blob），检索时纯 Python 暴力余弦（向量已归一化 → 点积）。
学习项目规模（数万 chunk）下完全够用且可观察全过程。
规模化时可平替 sqlite-vec / FAISS，接口不变。

表结构：
  chunks(id INTEGER PK, kind, cid, name, section, faq_idx, text, vec BLOB)
  index_meta(key, value)  建库时记 backend/model/dim/chunker，加载时据此构造匹配 embedder
"""
from __future__ import annotations
import json
import sqlite3
import struct
import math
from pathlib import Path
from typing import Iterable

from .chunker import Chunk


def _vec_to_blob(vec: list[float]) -> bytes:
    # float32 紧凑存储
    return struct.pack(f"<{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


class VectorIndex:
    def __init__(self, db_path: str | Path, dim: int | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：允许 Web 请求线程复用主线程建的连接
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_schema()
        self._dim = dim

    def _init_schema(self) -> None:
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  kind TEXT, cid INTEGER, name TEXT, section TEXT, faq_idx INTEGER,"
            "  text TEXT, vec BLOB, extra TEXT"
            ");"
            "CREATE INDEX IF NOT EXISTS idx_kind ON chunks(kind);"
            "CREATE INDEX IF NOT EXISTS idx_cid ON chunks(cid);"
            "CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT);"
        )
        self.db.commit()

    # ---- 索引元数据：记录建库用的后端/模型/维度，供加载时构造匹配 embedder ----
    def set_meta(self, key: str, value) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO index_meta(key, value) VALUES(?, ?)",
            (key, "" if value is None else str(value)),
        )
        self.db.commit()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute(
            "SELECT value FROM index_meta WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else default

    def meta_all(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in
                self.db.execute("SELECT key, value FROM index_meta")}

    @property
    def dim(self) -> int:
        if self._dim is None:
            row = self.db.execute("SELECT vec FROM chunks WHERE vec IS NOT NULL LIMIT 1").fetchone()
            if row:
                self._dim = len(_blob_to_vec(row["vec"]))
            else:
                self._dim = 0
        return self._dim

    def count(self, kind: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM chunks"
        if kind:
            sql += " WHERE kind=?"
            return self.db.execute(sql, (kind,)).fetchone()[0]
        return self.db.execute(sql).fetchone()[0]

    def all_chunk_texts(self) -> list[str]:
        """取出全部 chunk 文本。TF-IDF 后端加载索引后需用它重新 fit 词表。"""
        return [r["text"] for r in self.db.execute("SELECT text FROM chunks")
                if r["text"]]

    def clear(self, kind: str | None = None) -> None:
        if kind:
            self.db.execute("DELETE FROM chunks WHERE kind=?", (kind,))
        else:
            self.db.execute("DELETE FROM chunks")
        self.db.commit()

    def indexed_cids(self, kind: str) -> set[int]:
        """该 kind 下已入库的 cid 集合（断点续传判重）。"""
        return {r[0] for r in
                self.db.execute("SELECT DISTINCT cid FROM chunks WHERE kind=? AND cid IS NOT NULL",
                                (kind,))}

    def add(self, chunks: Iterable[Chunk], vectors: list[list[float]]) -> int:
        chunks = list(chunks)
        assert len(chunks) == len(vectors), "chunk 与向量数量不一致"
        rows = []
        for c, v in zip(chunks, vectors):
            rows.append((
                c.meta.get("kind"), c.meta.get("cid"), c.meta.get("name"),
                c.meta.get("section"), c.meta.get("faq_idx"),
                c.text, _vec_to_blob(v),
                json.dumps({k: c.meta[k] for k in c.meta
                            if k not in ("kind", "cid", "name", "section", "faq_idx")},
                           ensure_ascii=False),
            ))
        self.db.executemany(
            "INSERT INTO chunks(kind,cid,name,section,faq_idx,text,vec,extra) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        self.db.commit()
        if rows and self._dim is None:
            self._dim = len(vectors[0])
        return len(rows)

    def search(self, qvec: list[float], k: int = 5, kind: str | None = None,
               filter_cid: int | None = None) -> list[tuple[Chunk, float]]:
        """暴力余弦（向量已归一化 → 点积）。返回 (chunk, score) 降序。"""
        where = ["vec IS NOT NULL"]
        params: list = []
        if kind:
            where.append("kind=?")
            params.append(kind)
        if filter_cid is not None:
            where.append("cid=?")
            params.append(filter_cid)
        sql = ("SELECT * FROM chunks WHERE " + " AND ".join(where))
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in self.db.execute(sql, params):
            v = _blob_to_vec(row["vec"])
            s = _dot(qvec, v)
            scored.append((s, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[tuple[Chunk, float]] = []
        for s, row in scored[:k]:
            extra = json.loads(row["extra"]) if row["extra"] else {}
            meta = {"kind": row["kind"], "cid": row["cid"], "name": row["name"],
                    "section": row["section"], "faq_idx": row["faq_idx"], **extra}
            out.append((Chunk(text=row["text"], meta=meta), float(s)))
        return out


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
