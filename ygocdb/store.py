"""本地卡池缓存与离线查询。

流程：拉 /cards.zip.md5 比对本地，变更才下载 cards.zip，解出 cards.json，
落地 sqlite 表 cards。支持纯本地按 名/id/cid 查询，供 RAG 与 Agent 复用。

为学习目的，下载/校验/入库的每一步都打印进度，便于观察"增量同步"。
"""
from __future__ import annotations
import json
import sqlite3
import threading
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .client import YgocdbClient


SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    cid        INTEGER PRIMARY KEY,
    id         INTEGER,
    cn_name    TEXT,
    sc_name    TEXT,
    md_name    TEXT,
    nwbbs_n    TEXT,
    cnocg_n    TEXT,
    jp_name    TEXT,
    en_name    TEXT,
    types      TEXT,
    pdesc      TEXT,
    desc       TEXT,
    atk        INTEGER,
    def        INTEGER,
    level      INTEGER,
    race       INTEGER,
    attribute  INTEGER,
    type       INTEGER,
    setcode    INTEGER,
    ot         INTEGER,
    faqcount   INTEGER DEFAULT 0,
    artid      INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cn_name ON cards(cn_name);
CREATE INDEX IF NOT EXISTS idx_id     ON cards(id);

CREATE TABLE IF NOT EXISTS details (
    cid INTEGER PRIMARY KEY,
    id  INTEGER,
    json TEXT,
    fetched_at TEXT
);
"""


def _row_to_card(row: sqlite3.Row) -> dict:
    d = dict(row)
    # 拼回与 API 一致的结构，方便复用 codec/agent
    d["data"] = {
        "ot": d.pop("ot"), "setcode": d.pop("setcode"), "type": d.pop("type"),
        "atk": d.pop("atk"), "def": d.pop("def"), "level": d.pop("level"),
        "race": d.pop("race"), "attribute": d.pop("attribute"),
    }
    d["text"] = {
        "types": d.pop("types"), "pdesc": d.pop("pdesc"), "desc": d.pop("desc"),
    }
    return d


class CardStore:
    def __init__(self, cache_dir: str | Path = "./cache",
                 base: str = "https://ygocdb.com"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.zip_path = self.cache_dir / "cards.zip"
        self.json_path = self.cache_dir / "cards.json"
        self.db_path = self.cache_dir / "cards.db"
        self.client = YgocdbClient(base=base, min_interval=0.3)
        self._db: sqlite3.Connection | None = None
        self._lock = threading.Lock()  # 写操作加锁（Web 多线程下）

    # ---- db ----
    def connect(self) -> sqlite3.Connection:
        if self._db is None:
            # check_same_thread=False：允许 FastAPI 请求线程复用主线程建的连接
            self._db = sqlite3.connect(self.db_path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._db.executescript(SCHEMA)
            self._db.commit()
        return self._db

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    # ---- 单卡详情（含 FAQ）带磁盘缓存 ----
    def fetch_detail(self, card_id: int, *, force: bool = False) -> dict | None:
        """按密码(id)拉取 show=all 完整信息，缓存到 details 表，避免重复请求站点。

        返回结构含 faqs/supplement/jppacks/enpacks/avail 等。
        """
        db = self.connect()
        if not force:
            row = db.execute("SELECT json FROM details WHERE id=?", (int(card_id),)).fetchone()
            if row:
                return json.loads(row["json"])
        detail = self.client.get_card(card_id, show_all=True)
        if detail is None:
            return None
        from datetime import datetime
        with self._lock:
            db.execute(
                "INSERT OR REPLACE INTO details(cid,id,json,fetched_at) VALUES(?,?,?,?)",
                (detail.get("cid"), int(card_id),
                 json.dumps(detail, ensure_ascii=False), datetime.now().isoformat()),
            )
            db.commit()
        return detail

    def fetch_faq_count(self, card_id: int) -> int:
        """获取某卡 FAQ 数（来自在线 show=all，带缓存）。"""
        d = self.fetch_detail(card_id)
        faqs = (d or {}).get("faqs") or []
        return len(faqs) if isinstance(faqs, list) else 0

    # ---- 同步 ----
    def _local_md5(self) -> str | None:
        p = self.cache_dir / "cards.zip.md5"
        return p.read_text(encoding="utf-8").strip() if p.exists() else None

    def refresh(self, force: bool = False) -> int:
        """按 MD5 增量同步卡池到 sqlite。返回入库卡数。"""
        print("[store] 拉取远端 MD5 …")
        remote_md5 = self.client.get_cards_zip_md5()
        local_md5 = self._local_md5()
        if not force and local_md5 == remote_md5 and self.db_path.exists():
            print(f"[store] MD5 一致（{remote_md5}），跳过下载。")
        else:
            print(f"[store] MD5 变更（{local_md5} -> {remote_md5}），下载 cards.zip …")
            self.client.download_cards_zip(str(self.zip_path))
            (self.cache_dir / "cards.zip.md5").write_text(remote_md5, encoding="utf-8")
            self._extract_json()
        count = self._load_into_db()
        print(f"[store] 完成，共 {count} 张卡。")
        return count

    def _extract_json(self) -> None:
        print("[store] 解压 cards.json …")
        with zipfile.ZipFile(self.zip_path) as z:
            with open(self.json_path, "wb") as f:
                f.write(z.read("cards.json"))

    def _load_into_db(self) -> int:
        print("[store] 载入 sqlite …")
        data: dict[str, Any] = json.loads(self.json_path.read_text(encoding="utf-8"))
        db = self.connect()
        rows = []
        for c in data.values():
            text = c.get("text", {}) or {}
            d = c.get("data", {}) or {}
            rows.append((
                c.get("cid"), c.get("id"), c.get("cn_name"), c.get("sc_name"),
                c.get("md_name"), c.get("nwbbs_n"), c.get("cnocg_n"),
                c.get("jp_name"), c.get("en_name"),
                text.get("types"), text.get("pdesc", ""), text.get("desc", ""),
                d.get("atk"), d.get("def"), d.get("level"),
                d.get("race"), d.get("attribute"), d.get("type"),
                d.get("setcode"), d.get("ot"),
                c.get("faqcount", 0), c.get("artid", 0),
            ))
        cols = [
            "cid", "id", "cn_name", "sc_name", "md_name", "nwbbs_n", "cnocg_n",
            "jp_name", "en_name", "types", "pdesc", "desc", "atk", "def",
            "level", "race", "attribute", "type", "setcode", "ot", "faqcount", "artid",
        ]
        placeholders = ",".join(["?"] * len(cols))
        db.executemany(
            f"INSERT OR REPLACE INTO cards ({','.join(cols)}) VALUES ({placeholders})",
            rows,
        )
        db.commit()
        return len(rows)

    # ---- 离线查询 ----
    def _query(self, sql: str, params: Iterable = ()) -> list[dict]:
        db = self.connect()
        cur = db.execute(sql, tuple(params))
        return [_row_to_card(r) for r in cur.fetchall()]

    def get_by_cid(self, cid: int) -> dict | None:
        rows = self._query("SELECT * FROM cards WHERE cid=?", (cid,))
        return rows[0] if rows else None

    def get_by_id(self, card_id: int) -> dict | None:
        rows = self._query("SELECT * FROM cards WHERE id=?", (int(card_id),))
        return rows[0] if rows else None

    def search_by_name(self, name: str, limit: int = 20) -> list[dict]:
        """本地按名称模糊查询（多个译名字段均匹配）。"""
        like = f"%{name}%"
        sql = (
            "SELECT * FROM cards WHERE cn_name LIKE ? OR sc_name LIKE ? OR "
            "md_name LIKE ? OR jp_name LIKE ? OR en_name LIKE ? LIMIT ?"
        )
        return self._query(sql, (like, like, like, like, like, limit))

    def find_cards_in_text(self, text: str, limit: int = 3) -> list[dict]:
        """实体链接：找出哪些卡的名称（或其片段）出现在 text 中。

        用最长公共子串匹配，支持简称与系列名：
          - "米德拉什" 命中"神影依·米德拉什"（4字公共子串）
          - "具象天使" 命中"四军之具象天使"等（4字子串）
          - "青眼" 命中"青眼白龙"等（2字子串）
        整名命中加权最高；按公共子串长度降序，避免短噪声。
        全表 14295 张，difflib 毫秒级。
        """
        from difflib import SequenceMatcher
        db = self.connect()
        rows = db.execute("SELECT cid, id, cn_name, sc_name FROM cards").fetchall()
        matches: list[tuple[int, sqlite3.Row]] = []
        for r in rows:
            name = r["cn_name"] or r["sc_name"]
            if not name or len(name) < 2:
                continue
            if name in text:
                # 整名命中，加权最高
                matches.append((len(name) + 1000, r))
                continue
            m = SequenceMatcher(None, text, name).find_longest_match(
                0, len(text), 0, len(name))
            if m.size >= 2:
                matches.append((m.size, r))
        matches.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _, r in matches[:limit]:
            c = self.get_by_cid(r["cid"])
            if c:
                out.append(c)
        return out
