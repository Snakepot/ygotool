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
                # 跳过否定/限制句（不能/无法/不可…）——这是"不允许做某事"，不是该卡执行的动作
                if re.search(r"不能|无法|不可|不得|不可以|无法", seg):
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
                    "snippet": seg[:50].replace("\n", " "),
                    "clause_idx": idx,
                    "section": sec_name,
                    "reason": "uncertain" if not structured else "effect",
                })
    return out


# 兼容旧引用
CLAUSE_SPLIT = CLAUSE_MARK
