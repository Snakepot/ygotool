"""切分策略（★ RAG 学习点：切分方式如何影响召回）。

三种切分器输出统一的 Chunk(text, meta)，便于在同一向量库下对比效果：

1. CardChunker   —— 整卡效果文本作为 1 个 chunk（粒度粗，召回到整张卡）
2. EffectChunker —— 按 ①②③ / 【灵摆效果】【怪兽效果】 拆成多 chunk（粒度中，召回到单条效果）
3. FaqChunker     —— show=all 的 faqs Q/A 对每对 1 个 chunk（粒度细，裁定问答主力）

学习观察点：同一问题下，三种切分检索 top-k 的重叠度与命中边界不同，
examples/01_chunking_compare.py 可直观看差异。
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Chunk:
    text: str
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        t = self.text[:24].replace("\n", " ")
        return f"Chunk({t}… | meta={self.meta})"


_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(s: str) -> str:
    """去掉 HTML 标签并把 <br> 转换行。FAQ 原文带大量 <a>/<br>，会污染向量。"""
    if not s:
        return ""
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = _TAG_RE.sub("", s)
    return s.strip()


# 效果标记：①②③④⑤⑥⑦⑧⑨⑩⑪⑫ 等，以及「【...】」段标题
_EFFECT_MARK = re.compile(r"[①-⑫⑬-⑱]")
_SECTION_MARK = re.compile(r"【[^】]+】")


class _BaseChunker:
    name: str = "base"

    def _card_header(self, card: dict) -> str:
        text = card.get("text", {}) or {}
        name = card.get("cn_name") or card.get("name") or card.get("en_name") or "?"
        types = (text.get("types") or "").split("\n")[0]
        return f"{name}（{types}）"


class CardChunker(_BaseChunker):
    """整卡效果 = 1 chunk。把卡名+属性串+效果正文拼一起。"""

    name = "card"

    def chunk(self, card: dict) -> list[Chunk]:
        text = card.get("text", {}) or {}
        header = self._card_header(card)
        pdesc = text.get("pdesc") or ""
        desc = text.get("desc") or ""
        body = (f"{header}\n" + (f"【灵摆效果】{pdesc}\n" if pdesc else "") +
                (f"【怪兽效果】{desc}" if desc else "")).strip()
        if not body:
            return []
        return [Chunk(text=body, meta={"cid": card.get("cid"), "id": card.get("id"),
                                       "name": header, "kind": "card"})]


class EffectChunker(_BaseChunker):
    """按 ①②③ 与 【…】 段拆分。每个效果条目 1 个 chunk。"""

    name = "effect"

    def chunk(self, card: dict) -> list[Chunk]:
        text = card.get("text", {}) or {}
        header = self._card_header(card)
        out: list[Chunk] = []
        pdesc = text.get("pdesc") or ""
        desc = text.get("desc") or ""
        # 段命名：灵摆怪兽区分灵摆/怪兽效果；魔法陷阱用"效果"；普通怪兽用"怪兽效果"
        data = card.get("data", {}) or {}
        is_monster = bool((data.get("type") or 0) & 0x1)  # TYPE_MONSTER
        monster_label = "怪兽效果" if is_monster else "效果"
        sections = []
        if pdesc:
            sections.append(("灵摆效果", pdesc))
        if desc:
            sections.append((monster_label, desc))
        if not sections:
            return []
        for sec_name, sec_text in sections:
            parts = self._split_effects(sec_text)
            for idx, p in enumerate(parts):
                p = p.strip()
                if not p:
                    continue
                out.append(Chunk(
                    text=f"{header}「{sec_name}」{p}",
                    meta={"cid": card.get("cid"), "id": card.get("id"),
                          "name": header, "section": sec_name,
                          "effect_idx": idx, "kind": "effect"},
                ))
        return out

    @staticmethod
    def _split_effects(text: str) -> list[str]:
        """按 ①②③ 切分；若无标记则按句号/换行兜底。"""
        if _EFFECT_MARK.search(text):
            # 在标记前切分，保留标记在前一段
            parts = re.split(r"(?=[①-⑫⑬-⑱])", text)
            return [p for p in parts if p.strip()]
        # 兜底：按换行或句号
        return [p for p in re.split(r"[\n。]", text) if p.strip()]


class FaqChunker(_BaseChunker):
    """show=all 的 faqs Q/A 对，每对 1 个 chunk。裁定问答主力检索源。

    faqs 结构（来自 /api/v0/card/:id?show=all）：
      [{"q": "...", "a": "..."}, ...] 或带其它字段名，这里做宽松解析。
    """
    name = "faq"

    def chunk(self, card: dict) -> list[Chunk]:
        faqs = card.get("faqs") or []
        if not isinstance(faqs, list):
            return []
        header = self._card_header(card)
        out: list[Chunk] = []
        for i, qa in enumerate(faqs):
            if not isinstance(qa, dict):
                continue
            # FAQ 原文为日文、含 HTML 标签；剥离后更干净（跨语言检索见 embedder 说明）
            q = strip_html(qa.get("q") or qa.get("question") or "")
            a = strip_html(qa.get("a") or qa.get("answer") or "")
            if not (q or a):
                continue
            out.append(Chunk(
                text=f"{header}\n问：{q}\n答：{a}",
                meta={"cid": card.get("cid"), "id": card.get("id"),
                      "name": header, "faq_idx": i, "kind": "faq",
                      "q": q, "a": a},
            ))
        return out


CHUNKERS = {"card": CardChunker, "effect": EffectChunker, "faq": FaqChunker}


def get_chunker(name: str) -> _BaseChunker:
    if name not in CHUNKERS:
        raise ValueError(f"未知切分器: {name}（可选: {list(CHUNKERS)}）")
    return CHUNKERS[name]()


def chunk_cards(cards: Iterable[dict], chunker_name: str) -> list[Chunk]:
    """对一批卡用指定切分器切分，返回全部 chunk。"""
    chunker = get_chunker(chunker_name)
    out: list[Chunk] = []
    for c in cards:
        out.extend(chunker.chunk(c))
    return out
