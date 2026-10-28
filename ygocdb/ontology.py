"""裁定动作本体（★ §五 #2：把"送入墓地/离开场上"等裁定原因/来源结构化）。

PSCT 正文是自由文本，无结构化字段（§13 边界）。本模块定义一个**半精确 taxonomy**：
每条动作 = (key, label, 正则, source, reason)。tag_actions 在 PSCT 解析后对效果体打标，
使"作为 cost 丢弃手卡"vs"效果处理时丢弃手卡"这类裁定关键区分能被结构化检索，
而非靠子串猜测。

边界（§13）：身份类（种族/类型/星级）位掩码精确；动作类靠文本+PSCT，措辞差异需兜底；
原因/裁定类最不精确，仍需 FAQ+原则。本体只把"来源/cost/effect"这段结构化，不承诺裁定全对。
"""
from __future__ import annotations
import re

# 同义词归一（与 agent/tools.py 一致）：舍弃↔丢弃、送入墓地↔送去墓地
SYNS = {"舍弃": "丢弃", "丢弃": "舍弃", "送入墓地": "送去墓地",
        "送去墓地": "送入墓地", "取对象": "为对象", "为对象": "取对象",
        "手牌": "手卡", "手札": "手卡"}


from dataclasses import dataclass


@dataclass
class Action:
    key: str
    label: str
    patterns: list[str]            # 在效果体上搜索的正则（原始字符串，运行期编译）
    source: str = ""               # hand/deck/field/—（送墓来源）
    reason_default: str = ""       # 空表示由条款上下文（cost/effect）决定


# taxonomy：覆盖 §13 动作类 + §五 #2 原因细分。patterns 顺序无关，命中即打标。
# 注：效果体已由 parse_clauses 剥离 cost，故这里 patterns 默认描"效果处理时的动作"；
# cost 段的动作由 tag_actions 单独扫描标 reason="cost"。
ACTIONS: list[Action] = [
    Action("discard_hand", "丢弃/舍弃手卡",
           [r"丢弃", r"舍弃"], source="hand"),
    Action("send_hand_to_gy", "手牌送去墓地",
           [r"手[牌卡][^。]{0,20}送[^\n。]{0,4}墓地", r"从手[牌卡][^。]{0,20}送[^\n。]{0,4}墓地"],
           source="hand"),
    Action("send_deck_to_gy", "卡组送去墓地",
           [r"卡组[^。]{0,12}送[^\n。]{0,4}墓地", r"从卡组[^。]{0,12}送[^\n。]{0,4}墓地"],
           source="deck"),
    Action("send_field_to_gy", "场上送去墓地",
           [r"场上[^。]{0,8}送[^\n。]{0,4}墓地", r"从场上[^。]{0,6}送[^\n。]{0,4}墓地"],
           source="field"),
    Action("send_to_gy", "送去墓地（来源未指明）",
           [r"送[^\n。]{0,4}墓地"], source=""),
    Action("destroy", "破坏", [r"破坏"], source=""),
    Action("release", "解放/献祭", [r"解放", r"献祭"], source=""),
    Action("banish", "除外", [r"除外", r"除[外去]"], source=""),
    Action("draw", "抽卡", [r"抽[^\n。]{0,2}卡"], source="deck"),
    Action("special_summon", "特殊召唤", [r"特殊召唤", r"特召"], source=""),
    Action("add_to_hand", "加入手卡/检索",
           [r"加入手[卡札]", r"加入手牌", r"从卡组.{0,6}加入手"], source="deck"),
    Action("leaves_field_by_opponent", "因对方效果离开场上",
           [r"对方[^。]{0,20}(离开|离|从)场上", r"因对方[^。]{0,20}场上",
            r"对方[^。]{0,20}(让|使|把).{0,6}(离开|离)场"],
           source="field", reason_default="opponent"),
]

# 预编译
_COMPILED: dict[str, list[re.Pattern]] = {
    a.key: [re.compile(p) for p in a.patterns] for a in ACTIONS
}
_BY_KEY: dict[str, Action] = {a.key: a for a in ACTIONS}


def zones_in_text(text: str) -> set[str]:
    """识别一段文本里出现的来源区（key 集合：hand/deck/gy/extra/field/banished）。

    只做词面识别：调用方通常只传"动作动词之前"的片段（来源区）。
    「卡组」用负向断言排除「额外卡组/融合卡组」。
    """
    out: set[str] = set()
    if not text:
        return out
    if "额外卡组" in text or "融合卡组" in text:
        out.add("extra")
    if re.search(r"(?<!额外)(?<!融合)卡组", text):
        out.add("deck")
    if "手卡" in text or "手牌" in text or "手札" in text:
        out.add("hand")
    if "墓地" in text:
        out.add("gy")
    if "场上" in text:
        out.add("field")
    if "除外区" in text or "除外" in text:
        out.add("banished")
    return out


def first_action_pos(text: str) -> int | None:
    """文本里最早出现的动作词位置（无则 None）。用于把"动作前的片段"当来源区。"""
    best: int | None = None
    for pats in _COMPILED.values():
        for pat in pats:
            m = pat.search(text or "")
            if m and (best is None or m.start() < best):
                best = m.start()
    return best


# 用于"来源区"的动词词表（动作词本身，而非含来源的整条正则）
VERB_WORDS = ["解放", "献祭", "丢弃", "舍弃", "送去墓地", "送入墓地", "送进墓地",
              "送去", "送入", "送进", "加入手卡", "加入手牌", "特殊召唤", "特召",
              "除外", "破坏", "抽卡", "抽"]


def first_verb_pos(text: str) -> int | None:
    """文本里最早出现的"动作动词"位置（无则 None）。比 first_action_pos 更准：
    不会把"从额外卡组…"里的"卡组"当成动作起点。"""
    if not text:
        return None
    best: int | None = None
    for w in VERB_WORDS:
        i = text.find(w)
        if i >= 0 and (best is None or i < best):
            best = i
    return best


def source_zones_from_phrase(phrase: str) -> set[str]:
    """从查询短语抽"来源区"：动词之前的片段；若动词在句首（如"解放手卡"），取全句区域。"""
    p = phrase or ""
    pos = first_verb_pos(p)
    if pos is None:
        return zones_in_text(p)
    z = zones_in_text(p[:pos])
    if z:
        return z
    return zones_in_text(p)


# ---- 来源（zone）抽取（PSCT：动作的来源/目的地）----
# 每个 zone 的中文措辞；用于把"从X把…动作"的 X 识别出来
SOURCES: list[tuple[str, list[str]]] = [
    ("extra", ["额外卡组", "融合卡组"]),
    ("deck", ["卡组"]),
    ("hand", ["手卡", "手牌", "手札"]),
    ("gy", ["墓地"]),
    ("field", ["场上", "场上表侧", "场上里侧"]),
    ("banished", ["除外区", "除外状态", "除外里侧"]),
]
# 按长度降序匹配，避免"卡组"误命中"额外卡组"里的"卡组"——故 extra 放前且先匹配


def extract_source(segment: str) -> str:
    """从一段效果文本识别动作来源 zone。返回 zone key（deck/hand/gy/extra/field/banished）或空。
    注：半精确——只看 zone 词出现，不严格绑定到动作（多动作段落会取首个命中的 zone）。
    """
    for key, words in SOURCES:
        for w in words:
            if w in segment:
                return key
    return ""


# zone 词 → key（供 extract_object_source 反查）
_ZONE2KEY: dict[str, str] = {w: k for k, ws in SOURCES for w in ws}
# 取对象/目标获取句式：zone + 数量 + 「目标」（PSCT：动作的来源绑在目标前）
_OBJ_SRC_RE = re.compile(
    r"(额外卡组|融合卡组|卡组|手卡|手牌|手札|墓地|场上|除外区|除外)[^「]{0,6}「")


def extract_object_source(clause: str) -> str:
    """识别"取目标"的来源 zone（zone+数量+「目标」模式），比 extract_source 精确：
    排除触发条件里的 zone（如"从额外卡组离开"），只取绑在目标前的 zone（如"墓地1只「阿不思」"）。
    """
    m = _OBJ_SRC_RE.search(clause or "")
    if m:
        return _ZONE2KEY.get(m.group(1), "")
    return ""


def extract_targets(segment: str) -> list[str]:
    """抽取一段效果里的目标卡名/系列 token：「...」书名号内的内容。
    这些是动作的显式目标（如「阿不思的落胤」「教导」）。"""
    import re
    return [t for t in re.findall(r"「([^」]+)」", segment) if len(t) >= 2]


# 动作单元分割连词：PSCT 用连词连接多个动作单元
# （那之后=then, 并且/以及=and, 此外=also, 然后/之后 等）
_CONJ = re.compile(r"那之后|或者|或|并且|以及|此外|然后|之后|且|，|。")


def split_action_segments(body: str) -> list[str]:
    """把一个效果体按连词切成动作单元段。每个段≈一个独立动作。"""
    return [s.strip() for s in _CONJ.split(body) if s.strip()]


def all_actions() -> list[Action]:
    return ACTIONS


def match_actions(text: str) -> list[str]:
    """文本里命中的 action key 列表（去重，保序）。"""
    out: list[str] = []
    for a in ACTIONS:
        if any(p.search(text) for p in _COMPILED[a.key]):
            out.append(a.key)
    return out


def _norm(phrase: str) -> str:
    """同义词归一：舍弃→丢弃、送入墓地→送去墓地，便于按动作词匹配。"""
    s = phrase
    for k, v in SYNS.items():
        if k in s:
            s = s.replace(k, v)
    return s


def map_phrase_to_actions(phrase: str) -> list[str]:
    """把用户自然短语映射到 canonical 动作 key（供 search_by_action 路由）。

    例：「舍弃手卡」→[discard_hand]、「从卡组送去墓地」→[send_deck_to_gy]、
        「因对方离开场上」→[leaves_field_by_opponent]。命中多个则都返回。
    """
    if not phrase:
        return []
    p = _norm(phrase)
    hits: list[str] = []
    # 来源+动作组合优先精确匹配
    if ("手" in p or "手牌" in p) and ("丢弃" in p or "舍弃" in p):
        hits.append("discard_hand")
    if "卡组" in p and "墓地" in p:
        hits.append("send_deck_to_gy")
    if ("手" in p) and "墓地" in p:
        hits.append("send_hand_to_gy")
    if "场上" in p and "墓地" in p and "手" not in p and "卡组" not in p:
        hits.append("send_field_to_gy")
    if "对方" in p and ("离开" in p or "场上" in p):
        hits.append("leaves_field_by_opponent")
    if "破坏" in p:
        hits.append("destroy")
    if "解放" in p or "献祭" in p or "祭品" in p:
        hits.append("release")
    if "除外" in p or "除开" in p:
        hits.append("banish")
    if "抽" in p and "卡" in p:
        hits.append("draw")
    if "加入手卡" in p or "加入手牌" in p or "检索" in p:
        hits.append("add_to_hand")
    if "特殊召唤" in p or "特召" in p:
        hits.append("special_summon")
    if "丢弃" in p and not hits:
        hits.append("discard_hand")
    if "墓地" in p and not hits:
        hits.append("send_to_gy")
    # 去重保序
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out
