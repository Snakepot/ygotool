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
        clean = re.sub(r"[^一-鿿A-Za-z0-9]", "", keyword or "")
        if len(clean) < 1 and not (source or target):
            return [], []

        # 来源归一（支持「手卡·场上」等组合写法）
        src_zone = _zone_key(source)

        # 结构化复合路径：有 source 或 target 就按动作单元匹配
        if src_zone or target:
            action_keys = set(map_phrase_to_actions(keyword)) if clean else set()
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
                entry = {"id": r["id"], "cn_name": r["cn_name"],
                         "types": ((r["types"] or "").splitlines() or [""])[0],
                         "snippet": hit_unit["snippet"]}
                if is_structured(desc) or is_structured(pdesc):
                    modern.append(entry)
                else:
                    uncertain.append(entry)
            return modern, uncertain

        # 退化路径：单关键词子串匹配（无 source/target）
        clean = re.sub(r"[^一-鿿A-Za-z0-9]", "", keyword or "")
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
            if not any(_has_active(b, action_terms) for b in bodies):
                continue
            entry = {"id": r["id"], "cn_name": r["cn_name"],
                     "types": ((r["types"] or "").splitlines() or [""])[0]}
            if is_structured(desc) or is_structured(pdesc):
                modern.append(entry)
            else:
                uncertain.append(entry)
        return modern, uncertain

    def search_by_action(self, phrase: str, reason: str | None = None,
                         source: str = "") -> tuple[list[dict], list[dict]]:
        """按裁定动作本体全量搜（§五 #2，search_by_effect 的结构化升级版）。

        把用户短语映射到 canonical 动作（discard_hand/send_deck_to_gy/...），
        用 tag_actions 给每张卡的效果打标，命中即收录。reason 过滤：
        - reason='cost'  只命中 cost 段（"作为cost丢弃"）
        - reason='effect' 只命中效果段（"效果处理时丢弃"）
        - None 按短语自检测（短语含 cost/代价→cost；含效果处理/作为效果→effect；否则都要）
        - source='手卡·场上' 只保留命中片段里含该区域词的（用于"从手卡解放"这类来源限定）
        opponent 轴（因对方离开场上）始终不受 cost/effect 过滤。
        每条结果带 snippet（命中的原文片段），供 LLM/前端展示"为何命中"，避免严格 grounding 下拒答。
        返回 (现代档, 未确认旧措辞档)。短语无法映射→([], [])，调用方回退 search_by_effect。
        """
        from ygocdb.ontology import map_phrase_to_actions
        from ygocdb.effect_parser import tag_actions, is_structured
        keys = map_phrase_to_actions(phrase)
        if not keys:
            return [], []
        p = phrase or ""
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

        db = self.store.connect()
        rows = db.execute(
            "SELECT cid,id,cn_name,sc_name,types,desc,pdesc FROM cards"
        ).fetchall()
        zone_words = _zone_words(source)
        modern: list[dict] = []
        uncertain: list[dict] = []
        for r in rows:
            desc = r["desc"] or ""
            pdesc = r["pdesc"] or ""
            tags = tag_actions(desc, pdesc)
            matched = [t for t in tags if t["action"] in keys and reason_ok(t["reason"])]
            if zone_words:
                # 组合来源（如 手卡·场上）要求条款里列出的区域词都出现；
                # 单一来源（如 手卡）只要求该词出现。
                matched = [t for t in matched
                           if all(w in (t.get("clause") or t["snippet"]) for w in zone_words)]
            if not matched:
                continue
            entry = {"id": r["id"], "cn_name": r["cn_name"],
                     "types": ((r["types"] or "").splitlines() or [""])[0],
                     "snippet": matched[0]["snippet"]}  # 命中片段，给 LLM 看依据
            if is_structured(desc) or is_structured(pdesc):
                modern.append(entry)
            else:
                uncertain.append(entry)
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
        clean = re.sub(r"[^一-鿿A-Za-z0-9]", "", phrase or "")
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
        SAMPLE = 15
        reason = args.get("reason")
        hdr = (f"按动作「{args.get('phrase','')}」" + (f"(仅{reason}段)" if reason else "")
               if name == "search_by_action" else f"按效果「{args.get('keyword','')}」")
        lines = [f"{hdr} 共匹配 {len(modern)+len(uncertain)} 张"
                 f"（现代{len(modern)}+未确认{len(uncertain)}），下面列前 {SAMPLE} 样例（完整名单见左栏搜索）："]
        if modern:
            lines.append("【现代】:")
            for c in modern[:SAMPLE]:
                snip = c.get("snippet", "") if name == "search_by_action" else ""
                lines.append(f"- {c['cn_name']}(#{c['id']}) {c['types']} | {snip}".rstrip())
        if uncertain:
            lines.append("【未确认(90X前旧措辞,须查FAQ)】:")
            for c in uncertain[:SAMPLE]:
                snip = c.get("snippet", "") if name == "search_by_action" else ""
                lines.append(f"- {c['cn_name']}(#{c['id']}) {c['types']} | {snip}".rstrip())
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
                        "可按 reason 过滤：cost=只查作为发动代价的（原文'才能发动'之前）；effect=只查效果处理时的。"
                        "可用 source 限定来源区域（如 手卡/场上/手卡·场上/卡组/墓地）。返回现代(已结构化)+未确认(90X前旧措辞)两档全量名单。",
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
