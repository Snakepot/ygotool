"""工具集（★ 学习点：function-calling 降幻觉）。

把数据层（store/client/codec/retriever）包装成"工具"：Agent 不靠记忆答卡，
而是调这些工具拉真实数据。这里定义：
  - ToolRuntime：可调用的工具实现（返回渲染好的文本，便于直接拼上下文）
  - TOOLS：OpenAI function-calling 的 JSON schema（接真模型时可启用原生工具调用）

设计为"工具返回文本"而非结构化对象，是因为当前 mock 后端不支持原生 tool-calling，
Agent 用确定性管线调用工具、把结果作为上下文喂给 LLM 综合——同样的降幻觉效果。
"""
from __future__ import annotations
import json
import re
from typing import Any

from config import Settings
from ygocdb.store import CardStore
from ygocdb.codec import decode_card
from rag.retriever import Retriever
from rag.chunker import Chunk
from .data_loader import apply_aliases


def _has_active(body: str, terms) -> bool:
    """body 是否含"主动"动作词（非被动触发语境）。

    排除：动作词前 12 字内含"被"（被...丢弃），或后 8 字内含 时/场合（...丢弃时/的场合）。
    只要有一处主动出现即命中。
    """
    for t in terms:
        start = 0
        while True:
            i = body.find(t, start)
            if i < 0:
                break
            before = body[max(0, i - 12):i]
            after = body[i + len(t):i + len(t) + 8]
            passive = ("被" in before) or ("时" in after) or ("场合" in after)
            if not passive:
                return True
            start = i + 1
    return False


def _fmt_decode(dec: dict) -> str:
    """把 decode_card 的输出（含 list/dict 字段）格式化成可读字符串，避免 ['光'] 这种渲染。"""
    def join(x):
        if isinstance(x, list):
            return "/".join(str(v) for v in x) if x else "—"
        return str(x) if x is not None else "—"
    level = dec.get("level")
    if isinstance(level, dict):
        if "link" in level:
            lv = f"Link-{level['link']}"
        elif "scales" in level:
            lv = f"刻{level['scales'][0]}/{level['scales'][1]} {level.get('level','?')}星"
        else:
            lv = f"{level.get('level','?')}星"
    else:
        lv = str(level) if level is not None else "—"
    return (f"属性{join(dec.get('attribute'))} 种族{join(dec.get('race'))} "
            f"类型{join(dec.get('type'))} {lv} ATK{dec.get('atk','—')} DEF{dec.get('def','—')} "
            f"setcode{join(dec.get('setcode'))}")


# ---- 来源（zone）词表与归一（search_by_effect / search_by_action 共用）----
SRC = {"卡组": "deck", "手卡": "hand", "手牌": "hand", "手札": "hand",
       "墓地": "gy", "额外卡组": "extra", "额外": "extra", "融合卡组": "extra",
       "场上": "field", "除外": "banished", "除外区": "banished"}


def _zone_tokens(source: str) -> list[str]:
    """把「手卡·场上」「手卡/墓地」等组合来源拆成 zone 词列表。"""
    s = re.sub(r"[·/、,，和及\s]+", " ", source or "")
    return [t for t in s.split(" ") if t]


def _zone_key(source: str) -> str:
    """来源短语 → 第一个可识别的 zone key（结构化检索用单 zone 语义）。"""
    for t in _zone_tokens(source):
        if t in SRC:
            return SRC[t]
    for k in SRC:
        if k and k in (source or ""):
            return SRC[k]
    return ""


def _zone_words(source: str) -> list[str]:
    """来源短语 → 原文片段里可搜索的 zone 词（search_by_action 的 snippet 过滤用）。"""
    return [t for t in _zone_tokens(source) if t in SRC]


def _zone_words_in_phrase(phrase: str) -> list[str]:
    """从自然语言短语里找区域词（如"解放手卡"→手卡、"从卡组送去墓地"→卡组）。"""
    out: list[str] = []
    for w in sorted(SRC, key=len, reverse=True):
        if w == "卡组":
            if re.search(r"(?<!额外)(?<!融合)卡组", phrase or ""):
                out.append(w)
        elif w in (phrase or ""):
            out.append(w)
    return out


def _query_zones(phrase: str) -> set[str]:
    """从查询短语抽"来源区"（动词之前的片段；动词在句首时取全句）。"""
    from ygocdb.ontology import source_zones_from_phrase
    return source_zones_from_phrase(_normalize_query(phrase))


def _zone_in_unit(word: str, text: str) -> bool:
    """zone 词是否出现在动作单元里。卡组要排除「额外卡组/融合卡组」（P04：卡组≠额外卡组）。"""
    if not word or not text:
        return False
    if word == "卡组":
        return re.search(r"(?<!额外)(?<!融合)卡组", text) is not None
    return word in text


def _matched_sentence(text: str, terms, maxlen: int = 120) -> str:
    """命中关键词所在的**完整分句原文**（按 。！？； 切），供 LLM 照抄当"来源/方式"依据。

    按"命中词权重 = 词长之和"选分句（长词优先），避免只挑到包含常见双字（如"效果"）
    的无关分句。没有片段可抄时 LLM 只能自拟概述，会编出原文没有的限定语——所以
    兜底检索也必须给原文片段。
    """
    sents = [s.strip() for s in re.split(r"(?<=[。！？；])", text or "") if s.strip()]
    best, best_score = "", 0
    for s in sents:
        score = sum(len(t) for t in terms if t in s)
        if score > best_score:
            best, best_score = s, score
    if best:
        return best[:maxlen]
    return (text or "").strip()[:maxlen]


_LEVEL_RE = re.compile(r"(\d+)星")
_ATTR_RE = re.compile(r"(光|暗|地|水|炎|风|神)属性")
_RACE_RE = re.compile(r"(兽战士|魔法师|鸟兽|爬虫类|念动力|幻神兽|创造神|电子界|幻想魔|海龙|战士|天使|恶魔|不死|机械|水生|岩石|植物|昆虫|雷|龙|兽|恐龙|鱼)族")
# 怪兽类型限定词（"通常怪兽/效果怪兽…"），用于特异度判定
_TYPE_WORDS = ["通常怪兽", "效果怪兽", "融合怪兽", "仪式怪兽", "同步怪兽",
               "超量怪兽", "连接怪兽", "灵摆怪兽", "调整怪兽", "二重怪兽",
               "同盟怪兽", "反转怪兽", "灵魂怪兽"]


def _normalize_query(text: str) -> str:
    """查询归一化：先按 aliases.md 扩写别名，再把 等级N / N星 / N星级 统一成「N星」。"""
    s = apply_aliases(text or "")
    s = re.sub(r"等级\s*(\d+)", r"\1星", s)
    s = re.sub(r"星级?\s*(\d+)", r"\1星", s)
    return s


def _query_modifiers(phrase: str) -> list[str]:
    """从用户短语里抽「限定条件」词（等级/属性/种族/怪兽类型），用于按特异度排序/分组（P03）。"""
    p = _normalize_query(phrase)
    mods: list[str] = []
    mods += [f"{m}星" for m in _LEVEL_RE.findall(p)]
    mods += [f"{m}属性" for m in _ATTR_RE.findall(p)]
    mods += [f"{m}族" for m in _RACE_RE.findall(p)]
    mods += [t for t in _TYPE_WORDS if t in p]
    seen, out = set(), []
    for m in mods:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _lcs_best(a: str, b: str) -> tuple[int, int]:
    """a 与 b 的最长公共连续子串：(长度, 在 b 中的起始位置)。用于"原文片段"检索打分。"""
    best, bpos = 0, 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b):
            if ca == cb:
                cur[j + 1] = prev[j] + 1
                if cur[j + 1] > best:
                    best, bpos = cur[j + 1], j + 1 - cur[j + 1]
        prev = cur
    return best, bpos


class ToolRuntime:
    """持有数据层句柄，提供工具方法。每个方法返回 (text, trace)。"""

    def __init__(self, store: CardStore, retriever: Retriever | None = None,
                 settings: Settings | None = None):
        self.store = store
        self.retriever = retriever
        self.settings = settings
        self._lim_cache: dict | None = None
        self._rel_map: dict | None = None

    # ---- 工具实现 ----
    def search_card(self, query: str, limit: int = 5) -> tuple[str, list[dict]]:
        hits = self.store.search_by_name(query, limit=limit)
        if not hits:
            return "未找到匹配卡牌。", []
        lines = []
        trace = []
        for h in hits:
            dec = decode_card(h["data"])
            types = h["text"]["types"].splitlines()[0]
            line = f"- {h['cn_name']}（id={h['id']}） {types} | {_fmt_decode(dec)}"
            lines.append(line)
            trace.append({"id": h["id"], "cn_name": h["cn_name"]})
        return "\n".join(lines), trace

    def get_card_detail(self, card_id: int) -> tuple[str, dict | None]:
        d = self.store.fetch_detail(card_id)
        if not d:
            return f"未找到 id={card_id}。", None
        text = d.get("text", {}) or {}
        faqs = d.get("faqs") or []
        out = (f"{d.get('cn_name')}（id={d.get('id')}）\n"
               f"{text.get('types','')}\n效果：{text.get('desc','')}")
        if faqs:
            out += f"\n官方FAQ数：{len(faqs)}"
        return out, {"id": d.get("id"), "faqcount": len(faqs)}

    def decode_card_fields(self, card_id: int) -> tuple[str, dict | None]:
        c = self.store.get_by_id(card_id)
        if not c:
            return f"未找到 id={card_id}。", None
        dec = decode_card(c["data"])
        return f"{c['cn_name']} 解码：{_fmt_decode(dec)}", dec

    def get_limits(self) -> tuple[str, dict]:
        lim = self._limits()
        parts = []
        for lang in ("cn", "ja", "en"):
            d = lim.get(lang, {})
            parts.append(f"{lang}: 禁{len(d.get('forbidden', d.get('banned', [])))} "
                         f"限{len(d.get('limited', []))} 准限{len(d.get('semiLimited', []))}")
        return "禁限卡表概览 → " + " | ".join(parts), lim

    def _limits(self) -> dict:
        """禁限卡表（带运行期缓存，避免反复请求）。"""
        if self._lim_cache is None:
            self._lim_cache = self.store.client.get_limits()
        return self._lim_cache

    def limit_status(self, card_id: int, lang: str = "cn") -> str:
        """查某卡在指定语种的禁限状态：禁卡/限制/准限制/无限制。"""
        d = self._limits().get(lang, {})
        id_ = int(card_id)
        forbidden = d.get("forbidden", d.get("banned", [])) or []
        if id_ in forbidden:
            return "禁卡"
        if id_ in (d.get("limited") or []):
            return "限制"
        if id_ in (d.get("semiLimited") or []):
            return "准限制"
        return "无限制"

    def get_release_date(self, card_id: int) -> str:
        """查某卡各语种首次发售日期（来自 /api/v0/releaseDates.json，带缓存）。
        用于判断是否为 2014 文本统一(PSCT)前的旧措辞卡。"""
        if self._rel_map is None:
            rd = self.store.client.get_release_dates()
            self._rel_map = {}
            for e in rd:
                if e.get("id") is not None:
                    self._rel_map[int(e["id"])] = e.get("release") or {}
        rel = self._rel_map.get(int(card_id)) or {}
        out = []
        for lang in ("jp", "en", "sc"):
            d = rel.get(lang) or {}
            if d:
                out.append(f"{lang}:{d.get('date', '?')}")
        return " / ".join(out) if out else "无发售信息"

    def search_by_text(self, keyword: str, limit: int = 20) -> tuple[str, list[dict]]:
        """按效果文本关键词在 cards.db 全量搜（SQL LIKE，本地、不联网）。

        用于"有哪些卡拥有X效果"类查询。处理三类措辞差异 + 语境打分：
          - 同义词（舍弃↔丢弃）、长词滑窗（魔法/陷阱 vs 魔法卡）、短词 2/3 字窗。
          - 语境打分：用户常问"发动后的效果"含某动作，故对"效果语境"（那之后/选/全部...）
            加分、"cost 语境"（...才能发动）减分，效果卡优先于 cost 卡。
        打分 = 最长匹配×3 + 效果语境加分 - cost 语境减分。
        """
        import re
        SYNS = {"舍弃": "丢弃", "丢弃": "舍弃", "送入墓地": "送去墓地",
                "送去墓地": "送入墓地", "取对象": "为对象", "为对象": "取对象"}
        # 效果语境：丢弃发生在效果处理时（非发动 cost）
        EFF = re.compile(r"那之后|全部丢弃|选.{0,12}丢弃|把那张卡丢弃|张手卡丢弃。|张卡丢弃。|丢弃。|抽.{0,12}丢弃")
        # cost 语境：丢弃作为发动代价
        COST = re.compile(r"丢弃.{0,10}才能发动|才能发动.{0,10}丢弃|手卡.{0,4}丢弃.{0,4}才能")

        clean = re.sub(r"[^一-鿿A-Za-z0-9]", "", keyword or "")
        if len(clean) < 2:
            return "关键词过短。", []
        variants = {clean}
        for k, v in SYNS.items():
            if k in clean:
                variants.add(clean.replace(k, v))
        cands: set[str] = set()
        for v in variants:
            cands.add(v)
            if len(v) >= 3:
                for i in range(len(v) - 2):
                    cands.add(v[i:i + 3])
            if len(v) >= 2:
                for i in range(len(v) - 1):
                    cands.add(v[i:i + 2])
        db = self.store.connect()
        # id -> (score, longest_match, row)
        best: dict[int, list] = {}
        for kw in sorted(cands, key=len, reverse=True):
            like = f"%{kw}%"
            # 不按 cid 截断：宽泛 2 字词也尽量多取，靠打分排序
            rows = db.execute(
                "SELECT cid,id,cn_name,sc_name,types,desc,pdesc FROM cards "
                "WHERE desc LIKE ? OR pdesc LIKE ? LIMIT ?",
                (like, like, 2000),
            ).fetchall()
            for r in rows:
                desc = (r["desc"] or "") + " " + (r["pdesc"] or "")
                eff = bool(EFF.search(desc))
                cost = bool(COST.search(desc))
                ctx = 10 if eff else (-5 if cost else 0)
                score = len(kw) * 3 + ctx
                if r["id"] not in best or score > best[r["id"]][0]:
                    best[r["id"]] = [score, len(kw), r]
        ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)
        cards = [b[2] for b in ranked[:limit]]
        if not cards:
            return f"未匹配到含「{keyword}」效果的卡。", []
        lines = []
        for r in cards:
            types = (r["types"] or "").splitlines()[0] if r["types"] else ""
            desc = (r["desc"] or "").replace("\n", " ")[:80]
            lines.append(f"- {r['cn_name']}(id={r['id']}) {types} | {desc}")
        return f"按效果文本「{keyword}」匹配到 {len(cards)} 张：\n" + "\n".join(lines), [
            {"id": r["id"], "cn_name": r["cn_name"]} for r in cards
        ]

    def search_by_series(self, question: str, limit: int = 300) -> tuple[list[dict], str]:
        """按卡名全量搜某系列：从问题里提取系列名(与卡名最长公共子串)，
        SQL LIKE 卡名，返回全部命中(不截断)。用于"X系列卡牌/所有X卡"类查询。"""
        from difflib import SequenceMatcher
        import re
        linked = self.store.find_cards_in_text(question, limit=3)
        if not linked:
            return [], ""
        # 用第一张匹配卡与问题求最长公共子串 → 系列名
        name = linked[0].get("cn_name") or linked[0].get("sc_name") or ""
        m = SequenceMatcher(None, question, name).find_longest_match(0, len(question), 0, len(name))
        token = question[m.a:m.a + m.size]
        # 去掉"系列/卡牌/的/等/还有/尽可能多/给出/信息"等样板词
        token = re.sub(r"[系列卡牌的等还有吗尽可能多给出信息？?\s]", "", token)
        if len(token) < 2:
            return [], token
        like = f"%{token}%"
        db = self.store.connect()
        rows = db.execute(
            "SELECT cid,id,cn_name,sc_name,types,desc,pdesc FROM cards "
            "WHERE cn_name LIKE ? OR sc_name LIKE ? LIMIT ?",
            (like, like, limit),
        ).fetchall()
        cards = [{"id": r["id"], "cn_name": r["cn_name"],
                  "types": ((r["types"] or "").splitlines() or [""])[0],
                  "desc": (r["desc"] or "").replace("\n", " ")[:80]} for r in rows]
        return cards, token

    def rag_search(self, query: str, k: int = 5, kind: str | None = None
                   ) -> tuple[list[tuple[Chunk, float]], list[dict]]:
        if self.retriever is None:
            return [], []
        hits = self.retriever.retrieve(query, k=k, kind=kind)
        return hits, [{"cid": c.meta.get("cid"), "name": c.meta.get("name"),
                       "score": round(s, 3)} for c, s in hits]

    def search_by_effect(self, keyword: str, source: str = "", target: str = ""
                        ) -> tuple[list[dict], list[dict]]:
        """按"效果体（非 cost）"全量搜，返回 (现代档, 未确认旧措辞档) 全部名单。

        复合查询（§五 精度修复）：传 source/target 时走**结构化 per-unit 匹配**——
        用 parse_action_units 把每条效果切成动作单元（按连词分），要求"动作∧来源∧目标"
        命中**同一单元**。例：从卡组特殊召唤阿不思→keyword=特殊召唤, source=卡组, target=阿不思的落胤。
        不传 source/target 时退化为单关键词子串匹配（向后兼容）。
        每条结果带 snippet（命中的动作单元原文片段）。
        """
        import re
        from ygocdb.effect_parser import effect_bodies, is_structured, parse_action_units
        from ygocdb.ontology import map_phrase_to_actions
        SYNS = {"舍弃": "丢弃", "丢弃": "舍弃", "送入墓地": "送去墓地",
                "送去墓地": "送入墓地", "取对象": "为对象", "为对象": "取对象"}
        clean = re.sub(r"[^一-鿿A-Za-z0-9]", "", _normalize_query(keyword or ""))
        if len(clean) < 1 and not (source or target):
            return [], []

        # 来源归一（支持「手卡·场上」等组合写法）
        src_zone = _zone_key(_normalize_query(source))

        # 结构化复合路径：有 source 或 target 就按动作单元匹配
        if src_zone or target:
            action_keys = set(map_phrase_to_actions(_normalize_query(keyword))) if clean else set()
            mods = _query_modifiers(" ".join([keyword or "", source or "", target or ""]))
            db = self.store.connect()
            rows = db.execute(
                "SELECT cid,id,cn_name,sc_name,types,desc,pdesc FROM cards"
            ).fetchall()
            modern, uncertain = [], []
            tgt = (target or "").strip()
            for r in rows:
                desc = r["desc"] or ""
                pdesc = r["pdesc"] or ""
                units = parse_action_units(desc, pdesc)
                hit_unit = None
                for u in units:
                    if action_keys and not (set(u["actions"]) & action_keys):
                        continue
                    if src_zone and u["source"] != src_zone:
                        continue
                    if tgt and tgt not in u["snippet"] and not any(tgt in t for t in u["targets"]):
                        continue
                    hit_unit = u
                    break
                if not hit_unit:
                    continue
                specific = bool(mods) and all(m in hit_unit["snippet"] for m in mods)
                entry = {"id": r["id"], "cn_name": r["cn_name"],
                         "types": ((r["types"] or "").splitlines() or [""])[0],
                         "snippet": hit_unit["snippet"], "specific": specific}
                if is_structured(desc) or is_structured(pdesc):
                    modern.append(entry)
                else:
                    uncertain.append(entry)
            # 特异度优先（P03），与 search_by_action 一致
            modern.sort(key=lambda c: 0 if c["specific"] else 1)
            uncertain.sort(key=lambda c: 0 if c["specific"] else 1)
            return modern, uncertain

        # 退化路径：单关键词子串匹配（无 source/target）
        clean = re.sub(r"[^一-鿿A-Za-z0-9]", "", _normalize_query(keyword or ""))
        if len(clean) < 2:
            return [], []
        action_terms: set[str] = set()
        for k, v in SYNS.items():
            if k in clean or v in clean:
                action_terms.add(k)
                action_terms.add(v)
        if not action_terms:
            action_terms.add(clean)
            if len(clean) >= 2:
                for i in range(len(clean) - 1):
                    action_terms.add(clean[i:i + 2])
        db = self.store.connect()
        rows = db.execute(
            "SELECT cid,id,cn_name,sc_name,types,desc,pdesc FROM cards"
        ).fetchall()
        modern: list[dict] = []
        uncertain: list[dict] = []
        for r in rows:
            desc = r["desc"] or ""
            pdesc = r["pdesc"] or ""
            bodies = effect_bodies(desc) + effect_bodies(pdesc)
            hit_body = next((b for b in bodies if _has_active(b, action_terms)), None)
            if hit_body is None:
                continue
            entry = {"id": r["id"], "cn_name": r["cn_name"],
                     "types": ((r["types"] or "").splitlines() or [""])[0],
                     # 兜底路径也给命中分句原文，避免 LLM 无依据自拟条件
                     "snippet": _matched_sentence(hit_body, action_terms)}
            if is_structured(desc) or is_structured(pdesc):
                modern.append(entry)
            else:
                uncertain.append(entry)
        return modern, uncertain

    def search_by_action(self, phrase: str, reason: str | None = None,
                         source: str = "") -> tuple[list[dict], list[dict]]:
        """按裁定动作本体全量搜（§五 #2，search_by_effect 的结构化升级版）。

        把用户短语映射到 canonical 动作（discard_hand/send_deck_to_gy/...），
        用 tag_action_units 按「动作单元」打标（来源/目标只在同一单元内绑定），命中即收录。
        reason 过滤：
        - reason='cost'  只命中 cost 段（"作为cost丢弃"）
        - reason='effect' 只命中效果段（"效果处理时丢弃"）
        - None 按短语自检测（短语含 cost/代价→cost；含效果处理/作为效果→effect；否则都要）
        - source='手卡·场上' 只保留同一动作单元里含该区域词的（用于"从手卡解放"这类来源限定）；
          source 留空时自动从短语里取区域词（"从卡组…"→卡组）。
        ★ 特异度（P03）：短语里的限定条件（等级3 / 3星 / 暗属性 / 龙族…）会标到命中的卡上，
          结果按 specific 优先排序——明确限定用户条件的卡排最前。
        每条结果带 snippet（命中的原文片段），供 LLM/前端展示"为何命中"，避免严格 grounding 下拒答。
        返回 (现代档, 未确认旧措辞档)。短语无法映射→([], [])，调用方回退 search_by_effect。
        """
        from ygocdb.ontology import map_phrase_to_actions
        from ygocdb.effect_parser import tag_action_units, is_structured
        p = _normalize_query(phrase or "")
        keys = map_phrase_to_actions(p)
        if not keys:
            return [], []
        if reason is None:
            want_cost = any(w in p for w in ("cost", "代价", "作为cost"))
            want_effect = any(w in p for w in ("效果处理", "作为效果", "效果时")) and not want_cost
        else:
            want_cost = reason == "cost"
            want_effect = reason == "effect"

        def reason_ok(r: str) -> bool:
            if r == "opponent":
                return True
            if not want_cost and not want_effect:
                return True
            if want_cost and r == "cost":
                return True
            if want_effect and r == "effect":
                return True
            return False

        q_zone_keys: set[str] = set()
        if source:
            q_zone_keys = {SRC[w] for w in _zone_words(_normalize_query(source))}
        if not q_zone_keys:
            q_zone_keys = _query_zones(p)
        mods = _query_modifiers(p)

        db = self.store.connect()
        rows = db.execute(
            "SELECT cid,id,cn_name,sc_name,types,desc,pdesc FROM cards"
        ).fetchall()
        modern: list[dict] = []
        uncertain: list[dict] = []
        for r in rows:
            desc = r["desc"] or ""
            pdesc = r["pdesc"] or ""
            tags = tag_action_units(desc, pdesc)
            matched = [t for t in tags if t["action"] in keys and reason_ok(t["reason"])]
            if q_zone_keys:
                # 来源区：查询来源必须是该动作单元来源区的子集（来源更宽的卡进"次精确"）
                matched = [t for t in matched
                           if q_zone_keys <= set(t.get("zones") or [])]
            if not matched:
                continue
            t = matched[0]
            # 精确档位：来源一致(2) / 来源更宽(1) / 通用(0)，并与限定词档取较小值
            mod_units = [u for u in matched
                         if all(m in (u.get("clause") or "") for m in mods)]
            mod_ok = (not mods) or bool(mod_units)
            if q_zone_keys:
                uzones = set(mod_units[0].get("zones") or []) if mod_units else set()
                zone_tier = (2 if uzones == q_zone_keys
                             else (1 if q_zone_keys < uzones else 0))
            else:
                zone_tier = 2
            mod_tier = 2 if mod_ok else 0
            tier = min(zone_tier, mod_tier) if mods else zone_tier
            entry = {"id": r["id"], "cn_name": r["cn_name"],
                     "types": ((r["types"] or "").splitlines() or [""])[0],
                     "snippet": t["snippet"], "tier": tier,
                     "specific": tier == 2, "near": tier == 1,
                     "reason": t["reason"]}
            if is_structured(desc) or is_structured(pdesc):
                modern.append(entry)
            else:
                uncertain.append(entry)
        # 特异度优先：精确(2) > 次精确(1) > 通用(0)（P03：精确匹配 > 泛泛满足）
        modern.sort(key=lambda c: -c["tier"])
        uncertain.sort(key=lambda c: -c["tier"])
        return modern, uncertain

    def search_effect_phrase(self, phrase: str, top: int = 20
                             ) -> tuple[list[dict], str]:
        """按「效果原文片段」全量找卡（用户直接粘贴/引用原文措辞时用）。

        与词条/子串匹配不同：把原文与效果文本都做同样规范化（去标点/空白）后，
        按 **最长连续匹配** 打分（≥4 连续字），且 cost 段也算进被匹配文本
        （"才能发动"之前的解放/丢弃是原文的一部分）——这样
        「把自己的手卡·场上1只怪兽解放才能发动」这类整句粘贴也能找到卡。
        同分时按 4 字片段覆盖率二排，精确措辞的卡排最前。
        返回按匹配长度降序的 top N，每条带命中片段，供 LLM 核对"原文确实这么写"。
        """
        clean = re.sub(r"[^一-鿿A-Za-z0-9]", "", _normalize_query(phrase or ""))
        if len(clean) < 3:
            return [], clean
        grams3 = {clean[i:i + 3] for i in range(len(clean) - 2)}
        grams4 = [clean[i:i + 4] for i in range(len(clean) - 3)]
        db = self.store.connect()
        rows = db.execute(
            "SELECT id, cn_name, types, desc, pdesc FROM cards").fetchall()
        scored: list[tuple[int, int, int, Any, str]] = []
        for r in rows:
            body = re.sub(r"[^一-鿿A-Za-z0-9]",
                          "", (r["desc"] or "") + (r["pdesc"] or ""))[:400]
            if len(body) < 4 or not any(g in body for g in grams3):
                continue
            best, pos = _lcs_best(clean, body)
            if best < 4:
                continue
            cnt4 = sum(1 for g in grams4 if g in body)   # 片段覆盖率（同分二排）
            scored.append((best, cnt4, pos, r, body))
        if not scored:
            return [], clean
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        cards = []
        for best, _cnt4, pos, r, body in scored[:max(1, top)]:
            start = max(0, pos - 6)
            snip = body[start:pos + best + 24]
            cards.append({"id": r["id"], "cn_name": r["cn_name"],
                          "types": ((r["types"] or "").splitlines() or [""])[0],
                          "snippet": f"…{snip}…"})
        return cards, clean

    def run(self, name: str, args: dict[str, Any]) -> Any:
        fn = getattr(self, name, None)
        if fn is None:
            return f"未知工具: {name}", []
        return fn(**args)

    def call(self, name: str, args: dict[str, Any]) -> tuple[str, dict]:
        """ReAct 统一入口：按名字调度工具，把各种返回形状渲染成 LLM 可读文本。
        返回 (text, trace)。大名单只给总数+前N样例（防超上下文），完整名单走前端面板。
        """
        try:
            r = self.run(name, args or {})
        except Exception as e:
            return f"[{name}] 执行失败: {e}", {"error": str(e)}
        # 各工具返回形状不一，分别渲染
        if name in ("search_card", "get_card_detail", "decode_card_fields",
                    "get_limits", "search_by_text"):
            text = r[0] if isinstance(r, tuple) else str(r)
            return text, {"ok": True}
        if name in ("limit_status", "get_release_date"):
            return str(r), {"ok": True}
        if name in ("search_by_action", "search_by_effect"):
            modern, uncertain = r
            return self._render_list(name, modern, uncertain, args), \
                   {"modern": len(modern), "uncertain": len(uncertain)}
        if name == "search_by_series":
            cards, token = r
            return self._render_series(cards, token), {"count": len(cards), "token": token}
        if name == "search_effect_phrase":
            cards, token = r
            return self._render_phrase(cards, token), {"count": len(cards), "token": token}
        if name == "rag_search":
            hits, tr = r
            return self._render_rag(hits), {"hits": len(hits)}
        # 兜底
        return (r[0] if isinstance(r, tuple) else str(r)), {"ok": True}

    @staticmethod
    def _render_list(name: str, modern: list, uncertain: list, args: dict) -> str:
        SAMPLE = 10
        reason = args.get("reason")
        if name == "search_by_action":
            phrase = args.get("phrase", "")
            hdr = f"按动作「{phrase}」" + (f"(仅{reason}段)" if reason else "")
        else:
            phrase = " ".join([args.get("keyword", ""), args.get("source", ""),
                               args.get("target", "")]).strip()
            hdr = f"按效果「{args.get('keyword','')}」"
        mods = _query_modifiers(phrase)
        # 查询来源区（用于"来源一致/更宽"分组；search_by_action 才做）
        qz: set[str] = set()
        if name == "search_by_action":
            src = args.get("source", "")
            if src:
                from ygocdb.ontology import zones_in_text
                qz = zones_in_text(_normalize_query(src))
            if not qz:
                qz = _query_zones(phrase)
        zone_label = {"hand": "手卡", "deck": "卡组", "gy": "墓地",
                      "extra": "额外卡组", "field": "场上", "banished": "除外"}
        zone_text = "、".join(zone_label.get(z, z) for z in sorted(qz))
        total = len(modern) + len(uncertain)
        lines = [f"{hdr} 共匹配 {total} 张（现代{len(modern)}+未确认{len(uncertain)}）。"]

        def tier(c: dict) -> int:
            return c.get("tier", 2 if c.get("specific") else 0)

        if mods or qz:
            # 名单不大（≤60 张）时全量渲染：让 LLM 能完整列出，避免"只答前 30 张"与评测金标错位；
            # 上百张的大名单仍截断，完整名单走左栏浏览面板（控制 prompt token）。
            FULL = total <= 60
            SPEC_SAMPLE = 60 if FULL else 30
            NEAR_SAMPLE = 60 if FULL else 15
            gen_n = 60 if FULL else 10
            spec = [c for c in modern + uncertain if tier(c) == 2]
            near = [c for c in modern + uncertain if tier(c) == 1]
            gen_m = [c for c in modern if tier(c) == 0]
            gen_u = [c for c in uncertain if tier(c) == 0]
            if spec:
                if mods:
                    lines.append(f"★ 精确匹配（明确限定「{'、'.join(mods)}」且来源一致）：")
                else:
                    lines.append(f"★ 来源一致（来源限定为「{zone_text}」）：")
                for c in spec[:SPEC_SAMPLE]:
                    lines.append(f"- {c['cn_name']}(#{c['id']}) {c['types']} | {c.get('snippet','')}".rstrip())
            else:
                lines.append("（本批结果里没有来源/限定条件一致的卡）")
            if near:
                if mods:
                    lines.append(f"◆ 次精确（同样明确限定「{'、'.join(mods)}」，但来源更宽——可同时从手卡/卡组/场上等多个区域选取）：")
                else:
                    lines.append(f"◆ 来源更宽（包含「{zone_text}」，可同时从多个区域选取）：")
                for c in near[:NEAR_SAMPLE]:
                    lines.append(f"- {c['cn_name']}(#{c['id']}) {c['types']} | {c.get('snippet','')}".rstrip())
            if gen_m or gen_u:
                lines.append("【通用（未限定该条件，可自由选择对象/目标）】:")
                for c in gen_m[:gen_n]:
                    lines.append(f"- {c['cn_name']}(#{c['id']}) {c['types']} | {c.get('snippet','')}".rstrip())
                for c in gen_u[:gen_n]:
                    lines.append(f"- {c['cn_name']}(#{c['id']}) {c['types']} | {c.get('snippet','')}".rstrip())
            shown = (len(spec[:SPEC_SAMPLE]) + len(near[:NEAR_SAMPLE])
                     + len(gen_m[:gen_n]) + len(gen_u[:gen_n]))
            if total > shown:
                lines.append(f"（其余 {total - shown} 张见左栏完整名单）")
        else:
            SAMPLE = total if total <= 60 else 10
            if modern:
                lines.append("【现代】:")
                for c in modern[:SAMPLE]:
                    lines.append(f"- {c['cn_name']}(#{c['id']}) {c['types']} | {c.get('snippet','')}".rstrip())
            if uncertain:
                lines.append("【未确认(90X前旧措辞,须查FAQ)】:")
                for c in uncertain[:SAMPLE]:
                    lines.append(f"- {c['cn_name']}(#{c['id']}) {c['types']} | {c.get('snippet','')}".rstrip())
            if total > SAMPLE:
                lines.append(f"（其余 {total - SAMPLE} 张见左栏完整名单）")
        return "\n".join(lines)

    @staticmethod
    def _render_series(cards: list, token: str) -> str:
        SAMPLE = 15
        lines = [f"系列「{token}」共 {len(cards)} 张，下面列前 {SAMPLE} 样例（完整名单见左栏系列搜索）："]
        lines.append("、".join(f"{c['cn_name']}(#{c['id']})" for c in cards[:SAMPLE]))
        return "\n".join(lines)

    @staticmethod
    def _render_phrase(cards: list, token: str) -> str:
        SAMPLE = 15
        if not cards:
            return f"未找到原文含「{token}」的卡（按最长连续匹配≥4字）。"
        lines = [f"按原文片段「{token}」最长连续匹配排序，共 {len(cards)} 张（前 {SAMPLE}）："]
        for c in cards[:SAMPLE]:
            lines.append(f"- {c['cn_name']}(#{c['id']}) {c['types']} | {c['snippet']}")
        return "\n".join(lines)

    @staticmethod
    def _render_rag(hits: list) -> str:
        if not hits:
            return "（检索无相关片段）"
        lines = ["检索到的相关片段："]
        for ch, sc in hits:
            lines.append(f"[{round(sc,3)}] {ch.meta.get('name','?')} | {ch.text[:120]}")
        return "\n".join(lines)


# OpenAI function-calling 的 schema（ReAct：真模型按自然语言自选工具）
TOOLS = [
    {"type": "function", "function": {
        "name": "search_card",
        "description": "按卡名模糊查找某张卡（用户问'X是什么卡/叫什么/查X'时用）。返回卡名/密码/种类/属性/数值。",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "卡名关键词"},
                                      "limit": {"type": "integer", "default": 5, "description": "返回上限"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_card_detail",
        "description": "取某卡完整详情含效果文本与官方FAQ数量（裁定/详解某卡时用）。需先知道 card_id（密码）。",
        "parameters": {"type": "object",
                       "properties": {"card_id": {"type": "integer", "description": "卡牌密码 id"}},
                       "required": ["card_id"]}}},
    {"type": "function", "function": {
        "name": "search_by_action",
        "description": "按裁定动作全量找卡（用户问'有哪些/列出 cost包含X 的卡'，X是动作如 舍弃手卡/从卡组送去墓地/破坏/除外/解放/因对方离开场上）。"
                        "动作短语不必与原文逐字一致：'解放手卡怪兽' 与 '把自己的手卡·场上1只怪兽解放才能发动' 都归为动作 解放。"
        "**必须把用户全部限定词写进 phrase**（例：从卡组将等级3怪兽送入墓地 → phrase=从卡组把3星怪兽送去墓地），省略限定词会只得到通用卡；若返回里没有「★明确限定」分组，说明 phrase 漏了限定词，补上再查。"
                        "可按 reason 过滤：cost=只查作为发动代价的（原文'才能发动'之前）；effect=只查效果处理时的；不传 reason 时 cost 与效果段都搜（找「cost 包含X」的卡用它）。"
                        "可用 source 限定来源区域（如 手卡/场上/手卡·场上/卡组/墓地）；source 留空时自动从短语解析。结果分三档：★精确（限定一致+来源一致）/◆次精确（限定一致但来源更宽，如「手卡·卡组」）/通用；精确优先。返回现代(已结构化)+未确认(90X前旧措辞)两档全量名单。",
        "parameters": {"type": "object",
                       "properties": {"phrase": {"type": "string", "description": "动作短语，如 解放手卡 / 从卡组送去墓地 / 破坏"},
                                      "reason": {"type": "string", "enum": ["cost", "effect"],
                                                 "description": "过滤：cost=仅发动代价段，effect=仅效果处理段；不传则都要"},
                                      "source": {"type": "string",
                                                 "description": "来源区域限定（可选）：手卡/场上/手卡·场上/卡组/墓地/额外卡组/除外"}},
                       "required": ["phrase"]}}},
    {"type": "function", "function": {
        "name": "search_by_effect",
        "description": "按效果找卡。复合查询传 source/target 走结构化匹配（动作∧来源∧目标绑同一效果单元，精度高）："
                        "如『从卡组特殊召唤阿不思的落胤的卡』→ keyword=特殊召唤, source=卡组, target=阿不思的落胤。"
                        "source 可选值：卡组/手卡/墓地/额外卡组/场上/除外。target 是卡名片段。"
        "注意：只匹配效果体（「才能发动」之后）；cost 段（发动代价）不算——需要含 cost 的卡请用 search_by_action。"
        "结果同样按特异度排序：明确限定用户条件（等级/属性/种族/怪兽类型）的卡优先。"
        "不传 source/target 退化为单关键词子串匹配。返回现代+未确认两档名单（带命中片段）。",
        "parameters": {"type": "object",
                       "properties": {"keyword": {"type": "string", "description": "动作/效果关键词，如 特殊召唤/送去墓地/加入手卡"},
                                      "source": {"type": "string", "description": "来源 zone：卡组/手卡/墓地/额外卡组/场上/除外（可选，支持 手卡·场上 组合写法）"},
                                      "target": {"type": "string", "description": "目标卡名片段（可选，如 阿不思的落胤）"}},
                       "required": ["keyword"]}}},
    {"type": "function", "function": {
        "name": "search_effect_phrase",
        "description": "按「效果原文片段」找卡（用户直接粘贴/引用一整句原文措辞时优先用）。"
                        "按最长连续匹配打分（≥4 连续字），cost 段也参与匹配；返回命中片段供核对原文。",
        "parameters": {"type": "object",
                       "properties": {"phrase": {"type": "string",
                                                 "description": "效果原文片段，如 把自己的手卡·场上1只怪兽解放才能发动"},
                                      "top": {"type": "integer", "default": 20, "description": "返回上限"}},
                       "required": ["phrase"]}}},
    {"type": "function", "function": {
        "name": "search_by_series",
        "description": "按卡名全量找某系列所有卡（用户问'X系列/所有X卡'）。返回全部命中不截断。",
        "parameters": {"type": "object",
                       "properties": {"question": {"type": "string", "description": "含系列名的问句或系列名"}},
                       "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "rag_search",
        "description": "在卡牌效果/FAQ 语料库中语义检索与问题相关的片段（裁定问答主力：找官方FAQ与效果原文）。跨卡裁定组合问题用。",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "检索问句"},
                                      "k": {"type": "integer", "default": 5},
                                      "kind": {"type": "string", "enum": ["card", "effect", "faq"],
                                               "description": "限定检索的切分类型"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "decode_card_fields",
        "description": "把某卡的 YGOPro 位掩码字段解码为可读属性（种族/类型/星级/Link/灵摆刻度）。",
        "parameters": {"type": "object",
                       "properties": {"card_id": {"type": "integer"}},
                       "required": ["card_id"]}}},
    {"type": "function", "function": {
        "name": "get_limits",
        "description": "获取当前禁限卡表概览（禁/限/准限数量）。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "limit_status",
        "description": "查某卡在指定语种的禁限状态（禁卡/限制/准限制/无限制）。",
        "parameters": {"type": "object",
                       "properties": {"card_id": {"type": "integer"}, "lang": {"type": "string", "default": "cn"}},
                       "required": ["card_id"]}}},
    {"type": "function", "function": {
        "name": "get_release_date",
        "description": "查某卡各语种首次发售日期（判断是否 2014 PSCT 统一前的旧措辞卡）。",
        "parameters": {"type": "object",
                       "properties": {"card_id": {"type": "integer"}},
                       "required": ["card_id"]}}},
]
