"""学习示例 05：裁定动作本体（★ §五 #2：把"送入墓地/离开场上"原因结构化）。

演示两件事：
1. tag_actions 对典型卡打 cost/effect/uncertain 标签（魔力净化=cost、效果型=effect、旧措辞=uncertain）。
2. search_by_action（本体，按 canonical 动作 + reason 过滤）vs search_by_effect（子串）
   在同一查询下的命中数差异，量化"结构化检索能按 cost/effect 过滤"的价值。

跑法：python examples/05_action_ontology.py
（纯本地，无需 LLM Key；输出到 stdout 与 05_action_ontology.csv）
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from config import get_settings
from ygocdb.store import CardStore
from ygocdb.effect_parser import tag_actions, render_action_tags
from agent.tools import ToolRuntime


def part1_tags():
    print("=== Part1: tag_actions cost/effect 区分 ===")
    cases = [
        ("魔力净化(cost)", "①：丢弃1张手卡才能发动；选场上1张魔法陷阱卡破坏。"),
        ("效果型丢弃", "①：以自己墓地1只怪兽为对象才能发动；从手卡丢弃1只那只怪兽特殊召唤。"),
        ("卡组送墓", "①：从卡组把1只怪兽送去墓地。"),
        ("旧措辞(90X前)", "丢弃1张手卡。"),
    ]
    for name, t in cases:
        tags = tag_actions(t)
        print(f"- {name}: {render_action_tags(tags) or '（无动作标签）'}")


def part2_compare():
    print("\n=== Part2: search_by_action(本体) vs search_by_effect(子串) ===")
    settings = get_settings()
    st = CardStore(cache_dir=settings.cache_dir, base=settings.ygocdb_base)
    rt = ToolRuntime(st)
    queries = [
        "舍弃手卡",            # 通用：本体 vs 子串都该命中大量
        "作为cost丢弃手卡",     # 仅本体能按 reason=cost 过滤
        "从卡组送去墓地",      # 来源=deck
        "因对方效果离开场上",  # 原因=opponent，本体专用
    ]
    rows = []
    for q in queries:
        ma, ua = rt.search_by_action(q)
        me, ue = rt.search_by_effect(q)
        act_total = len(ma) + len(ua)
        eff_total = len(me) + len(ue)
        print(f"- 「{q}」: 本体={act_total}(现代{len(ma)}+未确认{len(ua)}) "
              f"子串={eff_total}(现代{len(me)}+未确认{len(ue)})")
        print(f"    本体样例: {[c['cn_name'] for c in (ma+ua)[:5]]}")
        rows.append({"query": q, "action_total": act_total, "effect_total": eff_total,
                     "action_modern": len(ma), "effect_modern": len(me)})
    st.close()
    out = Path(__file__).parent / "05_action_ontology.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query", "action_total", "effect_total",
                                         "action_modern", "effect_modern"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n已落盘: {out}")


if __name__ == "__main__":
    part1_tags()
    part2_compare()
