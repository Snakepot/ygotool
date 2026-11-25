"""PSCT 效果文本解析器（★ 核心功能：区分 cost 与效果体）。

游戏王 90X 系列卡包后的现代文本(PSCT)有结构：
  - 按 ①②③… 分条，【灵摆效果】/【怪兽效果】 分段
  - "才能发动" 是 cost 与效果体的分水岭：cost 在其前，效果体在其后

本解析器把 desc/pdesc 拆成条款，分离 cost 与 effect_body，使"效果(非cost)包含X"
能精准检索，而非靠正则猜测。90X 前无结构标记的旧措辞卡分不出来，归为"未确认"档。
"""
from __future__ import annotations
import re

# 效果条款编号 ①-⑫ 等
CLAUSE_MARK = re.compile(r"[①-⑫⑬⑭⑮⑯⑰]")
SECTION_MARK = re.compile(r"【[^】]+】")
ACTIVATION = "才能发动"

# 现代 PSCT 结构标记：有编号条款 或 有"才能发动"
_STRUCTURED = re.compile(r"[①-⑫⑬⑭⑮⑯⑰]|才能发动")

# 修饰语剔除（动作单元打标用）：「从X特殊召唤的…」是相对修饰语，不是该单元里送墓/除外等动作的来源。
# 例：哀悼傀儡「把最多有从额外卡组特殊召唤的对方场上的怪兽数量＋1张的…从卡组送去墓地」
# ——「额外卡组」只是「特殊召唤」的来源，真正送墓来源是「卡组」；不剔会把整句的动词位置
# 截在「特殊召唤」上，从而把「从额外卡组」误当送墓来源（query 从额外卡组送墓时召回噪音）。
# 只匹配带「的」的修饰句（「从额外卡组特殊召唤1只怪兽」是真实动作，不能剔）。
_SRC_MODIFIER_RE = re.compile(r"从[^，。；、]{1,10}?(?:特殊召唤|召唤)的")


def _strip_src_modifiers(text: str) -> str:
    """去掉「从X特殊召唤的…」修饰语，返回可用于动作识别/来源区提取的文本。"""
    return _SRC_MODIFIER_RE.sub("", text or "")


def is_structured(desc: str | None) -> bool:
    """该文本是否为现代 PSCT 结构（有编号条款或"才能发动"）。无结构→旧措辞，cost/效果分不出。"""
    return bool(_STRUCTURED.search(desc or ""))


def parse_clauses(desc: str | None) -> list[tuple[str, str]]:
    """拆成 [(clause_full, effect_body)]。

    effect_body = 去掉 cost 后的效果正文：
      - 含"才能发动"：取"才能发动。"之后为效果体（cost 在前，不搜）
      - 不含：整条作效果体（永续/触发/旧文本）
    """
    desc = desc or ""
    out: list[tuple[str, str]] = []
    for part in CLAUSE_SPLIT.split(desc):
        part = part.strip().lstrip("：:.")
        if not part:
            continue
        if ACTIVATION in part:
            idx = part.find(ACTIVATION)
            after = part[idx + len(ACTIVATION):].lstrip("。.：:")
            body = after.strip()
        else:
            body = part
        out.append((part, body))
    return out


def effect_bodies(desc: str | None) -> list[str]:
    """全部条款的效果体（cost 已剥离）。用于"效果(非cost)包含X"搜索。"""
    return [b for _, b in parse_clauses(desc) if b]


# ---- 动作打标（§五 #2：裁定动作本体）----
def _cost_body(clause: str) -> tuple[str, str]:
    """单条款里分离 cost 与效果体。返回 (cost_text, body_text)。
    cost = "才能发动"之前；body = 之后。无"才能发动"则 cost 空、整条作 body。
    """
    if ACTIVATION in clause:
        idx = clause.find(ACTIVATION)
        return clause[:idx], clause[idx + len(ACTIVATION):].lstrip("。.：:").strip()
    return "", clause.strip()


def tag_actions(desc: str | None, pdesc: str | None = None,
                section: str = "") -> list[dict]:
    """对一条卡的正文打动作标签（半精确，§13 边界）。

    复用 parse_clauses 的"才能发动"分离：cost 段的动作标 reason="cost"，
    效果体的动作标 reason="effect"。无结构标记（is_structured False）的旧措辞
    标 reason="uncertain"（对接 §13 未确认档，须查 FAQ）。

    返回 [{action, label, source, reason, clause_idx, section, snippet}]。
    """
    from .ontology import match_actions, _BY_KEY
    out: list[dict] = []
    sections: list[tuple[str, str]] = []
    if pdesc:
        sections.append(("灵摆效果", pdesc))
    if desc:
        sections.append(("怪兽效果" if not section else section, desc))
    for sec_name, text in sections:
        structured = is_structured(text)
        clauses = parse_clauses(text)
        for idx, (clause_full, body) in enumerate(clauses):
            cost_text, body_text = _cost_body(clause_full)
            # cost 段与效果体分别扫描
            for seg, reason in ((cost_text, "cost"), (body_text, "effect")):
                if not seg:
                    continue
                # 跳过否定/限制句（不能/无法/不可…）：那是"不允许做"，不是该卡执行的动作
                if re.search(r"不能|无法|不可|不得|不可以", seg):
                    continue
                for key in match_actions(seg):
                    a = _BY_KEY.get(key)
                    if not a:
                        continue
                    final_reason = a.reason_default or reason
                    if not structured:
                        final_reason = "uncertain"
                    out.append({
                        "action": key, "label": a.label, "source": a.source,
                        "reason": final_reason, "clause_idx": idx,
                        "section": sec_name,
                        "snippet": seg[:40].replace("\n", " "),
                        "clause": seg[:120].replace("\n", " "),
                    })
    # 去重（同 action+reason+clause 只留一条）
    seen, dedup = set(), []
    for t in out:
        k = (t["action"], t["reason"], t["clause_idx"], t["section"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(t)
    return dedup


def render_action_tags(tags: list[dict]) -> str:
    """把 tag_actions 结果渲染成给 ruling 上下文的紧凑标签串。"""
    if not tags:
        return ""
    parts = [f"{t['label']}({t['reason']})" for t in tags]
    return "[动作标签: " + "; ".join(parts) + "]"


# ---- 结构化动作单元解析（PSCT 范式：连词切单元 + 来源/目标绑定）----
def parse_action_units(desc: str | None, pdesc: str | None = None,
                        section: str = "") -> list[dict]:
    """把效果按 PSCT 范式解析成动作单元（★ §五 精度修复）。

    流程：parse_clauses 分条 + 剥 cost → 效果体按连词(那之后/并且/此外…)切成单元
    → 每单元抽 action(ontology 动作) / source(zone) / target(「卡名」)。

    半精确（§13）：能分清"送去墓地"与"加入手卡"是同条款内不同动作单元，
    且把"特殊召唤"动作绑到其来源/目标——避免 flat 子串把不同单元的词混在一起。
    无结构标记的旧卡 reason 标 uncertain。

    返回 [{actions, source, targets, snippet, clause_idx, section, reason}]。
    """
    from .ontology import (match_actions, _BY_KEY, extract_source,
                            extract_object_source, extract_targets,
                            split_action_segments)
    out: list[dict] = []
    sections: list[tuple[str, str]] = []
    if pdesc:
        sections.append(("灵摆效果", pdesc))
    if desc:
        sections.append(("怪兽效果" if not section else section, desc))
    # 代词指代（PSCT 常见：取对象句声明目标，动作句用"那只怪兽"指代）
    _ANAPHORA = re.compile(r"那只|这只|那张|这张|该怪兽|该卡|这只怪兽|那只怪兽")
    for sec_name, text in sections:
        structured = is_structured(text)
        clauses = parse_clauses(text)
        for idx, (clause_full, body) in enumerate(clauses):
            # clause 级目标/来源：取对象句（"以…为对象"在 cost 段）声明的，供代词句继承
            clause_targets = extract_targets(clause_full)
            # 用 extract_object_source 取"绑目标的 zone"（如 墓地1只「阿不思」），
            # 而非 extract_source（会误取触发词里的 zone，如"从额外卡组离开"→extra）
            clause_source = extract_object_source(clause_full) or extract_source(clause_full)
            # body 已剥 cost；按连词切单元
            for seg in split_action_segments(body):
                acts = match_actions(seg)
                if not acts:
                    continue
                # 否定/限制句（不能/无法/不可…）——这是"不允许做某事"，不是该卡执行的动作；
                # 只保留 cannot_activate（「不能把效果发动」本身是目标动作），其余动作丢弃
                if re.search(r"不能|无法|不可|不得|不可以", seg):
                    acts = [a for a in acts if a == "cannot_activate"]
                    if not acts:
                        continue
                src = extract_source(seg) or extract_object_source(seg)
                tgts = extract_targets(seg)
                # 代词句继承同 clause 的 target/source（动作在下一句、目标在取对象句）
                if _ANAPHORA.search(seg):
                    if not src:
                        src = clause_source
                    if not tgts:
                        tgts = clause_targets
                out.append({
                    "actions": acts,
                    "source": src,
                    "targets": tgts,
                    # 整段单元原文（含限定语），避免截断后丢条件
                    "snippet": seg[:120].replace("\n", " "),
                    "clause_idx": idx,
                    "section": sec_name,
                    "reason": "uncertain" if not structured else "effect",
                })
    return out


# 兼容旧引用
CLAUSE_SPLIT = CLAUSE_MARK

# 动作单元内的代词指代（PSCT 常见：取对象句声明目标，动作句用"那只怪兽"指代）
_ANAPHORA = re.compile(r"那只|这只|那张|这张|该怪兽|该卡|这只怪兽|那只怪兽")
# 否定/限制词：动作单元含这些词表示"不允许做"，不是该卡执行的动作
_NEG_RE = re.compile(r"不能|无法|不可|不得|不可以|不用|无需|不需要|不进行|未进行")
# 触发条件（发动条件）分句：PSCT 里 "…的场合"/"…时"/"…的时候"/"…之际"（可带「发动」/「才能发动」）
# 是条件，不是 cost / 效果处理。例：
#   森罗的守神「自己卡组的卡被效果送去墓地的场合，把这张卡1个超量素材取除才能发动」→ 场合=条件，取除=cost
#   缚神冢「场上的这张卡被效果破坏送去墓地时才能发动」→ 时=条件
#   缚神冢②「…的场合发动」→ 无「才能」的触发效果，场合=条件
_TRIGGER_RE = re.compile(r"(的)?(場合|场合|时|時|とき|时候|之际|情况下)([,，]?)(才能发动|发动)?$")
# cost 段里"祈使动作"句首（用于区分真 cost 与被动条件句）
_COST_LEAD_RE = re.compile(r"^(把|将|从|以|解放|献祭|丢弃|舍弃|支付|作为cost)")


def tag_action_units(desc: str | None, pdesc: str | None = None
                     ) -> list[dict]:
    """按 PSCT 动作单元打标（search_by_action 专用）：来源/目标只在同一动作单元内绑定。

    与 tag_actions 的区别：tag_actions 以"整条款"为单位打标，条款里任意位置出现 zone 词
    都可能被当成来源（如"那只怪兽解放，从手卡把…特殊召唤"中手卡属于特殊召唤，却会被误绑到
    解放）；本函数按连词（那之后/并且/此外/，/。）把 cost/效果体切成动作单元，zone/目标只在
    该单元内提取，并跳过否定/限制单元（不能/无法/不可/不得/不用/无需/不需要…）。
    cost 段标 reason=cost，效果体标 reason=effect；无 PSCT 结构的旧卡标 uncertain。
    返回 [{action,label,source,reason,clause_idx,section,snippet,clause,targets}]。
    """
    from .ontology import (match_actions, _BY_KEY, extract_source,
                            extract_object_source, extract_targets,
                            split_action_segments, zones_in_text,
                            first_verb_pos)
    out: list[dict] = []
    sections: list[tuple[str, str]] = []
    if pdesc:
        sections.append(("灵摆效果", pdesc))
    if desc:
        sections.append(("怪兽效果", desc))
    for sec_name, text in sections:
        structured = is_structured(text)
        clauses = parse_clauses(text)
        for idx, (clause_full, body) in enumerate(clauses):
            clause_targets = extract_targets(clause_full)
            # 代词指代句继承的目标来源：只认"区域+数量+「目标」"这种绑定句式的 zone，
            # 不退回 extract_source（否则会把条款里别处出现的区域词误当来源，如次元魔法）
            clause_source = extract_object_source(clause_full)
            cost_text, _body = _cost_body(clause_full)
            # 条款 cost 段的区域（供代词句继承来源区，如"以墓地「X」为对象…那只怪兽特殊召唤"）
            clause_zones = zones_in_text(cost_text)
            for seg, reason in ((cost_text, "cost"), (body, "effect")):
                if not seg:
                    continue
                for unit in split_action_segments(seg):
                    unit = unit.strip()
                    if not unit:
                        continue
                    # 先剔「从X特殊召唤的…」修饰语再识别动作/来源：
                    # 否则动词位置被修饰语里的「特殊召唤」截断，来源区错取成「从额外卡组」
                    clean = _strip_src_modifiers(unit) or unit
                    acts = match_actions(clean)
                    if not acts:
                        continue
                    # 否定/限制句（不能/无法/不用…＝"不允许做"，不是该卡执行的动作）：
                    # 只保留 cannot_activate（「不能把效果发动」本身是目标动作）；
                    # 其余动作（如「对方若不从卡组送去墓地则不能把魔法卡发动」里的送墓）是自肃/条件，丢弃
                    if _NEG_RE.search(unit):
                        acts = [a for a in acts if a == "cannot_activate"]
                        if not acts:
                            continue
                    # 触发条件分句：OCG 文本里「场合/場合」是条件句标记（不限于句尾，如
                    # 「把…解放的场合可以特殊召唤」这种特殊召唤条件），「…时/…之际/…的时候」
                    # 收尾也是条件。所有分句都做判定（无「才能发动」的句式同样适用）。
                    # cost 段再区分祈使句（真 cost：「把…丢弃」）与被动条件句（「…被…破坏」= 条件的一部分）。
                    unit_reason = reason
                    if (_TRIGGER_RE.search(unit) or "场合" in unit or "場合" in unit
                            or "时候" in unit or "之际" in unit):
                        unit_reason = "trigger"
                    elif reason == "cost":
                        if _COST_LEAD_RE.match(unit) or "作为cost" in unit:
                            unit_reason = "cost"
                        elif "被" in unit:
                            unit_reason = "trigger"
                        else:
                            unit_reason = "cost"
                    src = extract_object_source(clean) or extract_source(clean)
                    tgts = extract_targets(clean)
                    # 来源区集合：动词之前的片段；动词在句首时，解放/丢弃类取动词后的区域词
                    pos = first_verb_pos(clean)
                    if pos is not None:
                        zones = zones_in_text(clean[:pos])
                        if not zones and pos == 0 and any(
                                a in ("release", "discard_hand") for a in acts):
                            zones = zones_in_text(clean)
                    else:
                        zones = zones_in_text(clean)
                    if _ANAPHORA.search(unit):
                        if not src:
                            src = clause_source
                        if not tgts:
                            tgts = clause_targets
                        if not zones:
                            zones = clause_zones
                    for key in acts:
                        a = _BY_KEY.get(key)
                        if not a:
                            continue
                        out.append({
                            "action": key, "label": a.label,
                            "source": src or a.source,
                            "zones": sorted(zones),
                            "reason": "uncertain" if not structured else unit_reason,
                            "clause_idx": idx, "section": sec_name,
                            # 片段=整段动作单元原文（含「那只怪兽以外」这类限定语），
                            # 供 LLM 照抄，避免被截断后自拟条件
                            "snippet": unit[:120].replace("\n", " "),
                            "clause": unit[:120].replace("\n", " "),
                            "targets": tgts,
                        })
    return out
