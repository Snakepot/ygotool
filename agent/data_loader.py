"""原则库 / 判例库 加载器（★ §五 #5/#6：人工策展内容的脚手架）。

把人工编写的内容喂进 Agent 与评测：
  - principles.md  → 解析成原则列表，渲染进 ruling 系统提示（成文法）
  - cases.jsonl   → 判例集，双重用途：few-shot 注入（教 ReAct 格式）+ 评测判例（量化正确率）

格式见 agent/data/ 下的模板文件。纯标准库解析，无新依赖。
"""
from __future__ import annotations
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_principles(path: Path | str | None = None) -> list[dict]:
    """解析 principles.md：`## PID: 标题` 一段一条，bullet `- key: value` 取字段。
    多行值（如多个正例）允许多行续写，合并到上一字段。
    """
    p = Path(path) if path else (DATA_DIR / "principles.md")
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8")
    out: list[dict] = []
    cur: dict | None = None
    last_key: str | None = None
    for line in text.splitlines():
        ls = line.strip()
        if ls.startswith("## "):
            if cur:
                out.append(cur)
            head = ls[3:].strip()
            # "P01: cost 与效果的区分"
            pid, _, title = head.partition(":")
            cur = {"id": pid.strip(), "title": title.strip()}
            last_key = None
        elif cur is not None and ls.startswith("- "):
            body = ls[2:]
            key, _, val = body.partition(":")
            key, val = key.strip(), val.strip()
            cur[key] = val
            last_key = key
        elif cur is not None and ls and last_key and not ls.startswith("#"):
            # 续行：追加到上一字段（用于多正例/多反例）
            cur[last_key] = (cur.get(last_key, "") + " " + ls).strip()
    if cur:
        out.append(cur)
    return out


def render_principles_block(principles: list[dict] | None = None) -> str:
    """把原则列表渲染成拼进 ruling 系统提示的文本块。无原则时返回空串。"""
    if principles is None:
        principles = load_principles()
    if not principles:
        return ""
    lines = ["【裁定原则库】（人工策展，成文法；引用时用原则编号如 P01）"]
    for p in principles:
        parts = [f"{p.get('id','?')}: {p.get('title','')}"]
        if p.get("stmt"):
            parts.append(f"  规则：{p['stmt']}")
        if p.get("why"):
            parts.append(f"  缘由：{p['why']}")
        if p.get("applies"):
            parts.append(f"  适用：{p['applies']}")
        if p.get("正例"):
            parts.append(f"  正例：{p['正例']}")
        if p.get("反例"):
            parts.append(f"  反例：{p['反例']}")
        if p.get("src"):
            parts.append(f"  来源：{p['src']}")
        lines.append("\n".join(parts))
    return "\n".join(lines)


def load_cases(path: Path | str | None = None) -> list[dict]:
    """解析 cases.jsonl：一行一 JSON 判例。跳过空行/注释行。"""
    p = Path(path) if path else (DATA_DIR / "cases.jsonl")
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[data_loader] 跳过无法解析的行: {e}")
    return out


def fewshot_pairs(cases: list[dict] | None, task: str, n: int = 2
                  ) -> list[tuple[str, str]]:
    """从判例集挑 n 条可作 few-shot 的（fewshot=true 且 task 匹配）→ (q, ruling)。
    返回的 id 供 eval 排除（防泄漏）。"""
    if cases is None:
        cases = load_cases()
    picks = [c for c in cases if c.get("task") == task and c.get("fewshot")]
    return [(c["q"], c["ruling"]) for c in picks[:n]]


def fewshot_ids(cases: list[dict] | None, task: str, n: int = 2) -> set[str]:
    if cases is None:
        cases = load_cases()
    picks = [c for c in cases if c.get("task") == task and c.get("fewshot")]
    return {c["id"] for c in picks[:n]}


def eval_cases(cases: list[dict] | None, exclude_ids: set[str] | None = None
               ) -> list[dict]:
    """评测用判例集：排除当轮注入的 few-shot（防泄漏）。默认全部可用作评测。"""
    if cases is None:
        cases = load_cases()
    excl = exclude_ids or set()
    return [c for c in cases if c["id"] not in excl]
