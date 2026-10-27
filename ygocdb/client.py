"""ygocdb.com 官方 API 的轻量 HTTP 客户端（标准库实现，零依赖）。

端点（base 默认 https://ygocdb.com，均 GET、CORS=*，除 cardset 为 POST）：
  /api/v0/?search=:q      搜索卡牌（名/效果/密码/cid），分页 next
  /api/v0/card/:id        单卡详情；?show=all 返回 faqs/supplement/packs/avail
  /api/v0/cardset         批量取卡（POST JSON: ids/cids/names）
  /api/v0/cards.zip       全量卡池 zip
  /api/v0/cards.zip.md5   全量卡池 MD5
  /api/v0/limits.json     禁限卡表；?show=all 含历史
  /api/v0/releaseDates.json  发售信息
"""
from __future__ import annotations
import json
import time
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_UA = "ygotool/0.1 (+local learning project)"


class YgocdbClient:
    def __init__(self, base: str = "https://ygocdb.com", timeout: float = 30.0,
                 ua: str = DEFAULT_UA, min_interval: float = 0.0):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.ua = ua
        self._last = 0.0
        self.min_interval = min_interval  # 简单限流，避免高频打站点

    # ---- low-level ----
    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def _get(self, path: str, params: dict[str, Any] | None = None,
             stream: bool = False) -> Any:
        self._throttle()
        url = self.base + path
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": self.ua})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            if stream:
                return resp.read()
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, path: str, body: dict[str, Any]) -> Any:
        self._throttle()
        url = self.base + path
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"User-Agent": self.ua, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ---- endpoints ----
    def search(self, query: str, start: int = 0, callback: str | None = None) -> dict:
        """搜索卡牌。返回 {'result': [...], 'next': int}。next=0 表示无更多。"""
        params: dict[str, Any] = {"search": query, "start": start}
        if callback:
            params["callback"] = callback
        return self._get("/api/v0/", params)

    def search_all(self, query: str, limit: int = 200) -> list[dict]:
        """翻页拉取某个查询的全部结果（保护上限 limit）。"""
        out: list[dict] = []
        start = 0
        while True:
            page = self.search(query, start=start)
            out.extend(page.get("result", []))
            nxt = page.get("next", 0)
            if not nxt or len(out) >= limit:
                break
            start = nxt
        return out[:limit]

    def get_card(self, card_id: int, show_all: bool = False,
                 callback: str | None = None) -> dict:
        """按密码(id)取单卡。show_all=True 返回 faqs/supplement 等完整信息。"""
        params: dict[str, Any] = {}
        if show_all:
            params["show"] = "all"
        if callback:
            params["callback"] = callback
        return self._get(f"/api/v0/card/{int(card_id)}", params or None)

    def get_cardset(self, *, ids: list[int] | None = None,
                    cids: list[int] | None = None,
                    names: list[str] | None = None) -> dict:
        """批量取卡（POST JSON）。返回以请求键为名、卡信息为值的 dict。"""
        body: dict[str, Any] = {}
        if ids:
            body["ids"] = ids
        if cids:
            body["cids"] = cids
        if names:
            body["names"] = names
        return self._post_json("/api/v0/cardset", body)

    def get_limits(self, show_all: bool = False) -> dict:
        params = {"show": "all"} if show_all else None
        return self._get("/api/v0/limits.json", params)

    def get_release_dates(self) -> list:
        return self._get("/api/v0/releaseDates.json")

    def get_cards_zip_md5(self) -> str:
        resp = self._get("/api/v0/cards.zip.md5")
        # 返回的是带引号字符串
        return resp.strip('"') if isinstance(resp, str) else resp

    def download_cards_zip(self, dest_path: str) -> str:
        data = self._get("/api/v0/cards.zip", stream=True)
        with open(dest_path, "wb") as f:
            f.write(data)
        return dest_path
