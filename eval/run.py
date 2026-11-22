"""评测 harness（★ §五 #6：设计评估体系、量化效果）。

跑 agent/data/cases.jsonl 判例集 → 每条调 CardAgent.ask →
  - LLM-as-judge 判回答与 golden ruling 是否语义一致（裁定结论相同）
  - 核查 must_cite（原则编号 / 卡 id）是否出现在答案 → 引用率
出 csv + 汇总（正确率 / 引用率）。

跑法：python eval/run.py
（用真模型 GLM，每条 ~10k token；num_few_shots=0 防判例泄漏）
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config
from ygocdb.store import CardStore
from agent.llm import LLMClient, Message
from agent.budget import TokenBudgeter
from agent.agent import CardAgent
from agent.data_loader import load_cases, eval_cases, fewshot_ids

SETTINGS = config.get_settings()


def make_agent(store: CardStore, retriever) -> CardAgent:
    llm = LLMClient("openai", base_url=SETTINGS.llm_base_url,
                    api_key=SETTINGS.llm_api_key, model=SETTINGS.llm_model)
    return CardAgent(store, llm, retriever=retriever,
                     budgeter=TokenBudgeter(max_context=SETTINGS.default_max_context_tokens),
                     settings=SETTINGS)


def judge_correctness(judge_llm: LLMClient, question: str, expected: str,
                      actual: str) -> tuple[bool | None, str]:
    """LLM-as-judge：判断 actual 与 expected 裁定结论是否语义一致。
    返回 (True/False/None, 理由)；None 表示 judge 输出无法判定（记为 judge 异常，
    不混入正确率——旧实现 `"一致" in "不一致"` 恒真，会把"不一致"判成一致）。"""
    sys_p = ("你是游戏王裁定裁判。给定问题、标准答案、模型答案，判断两者【裁定结论】是否"
             "语义一致（能否触发/能否发动/胜负等结论相同即可，措辞差异不重要）。"
             "第一行只输出 YES 或 NO（不要输出其它内容），第二行起简述理由。")
    user = (f"问题：{question}\n标准答案：{expected}\n模型答案：{actual}\n"
            f"判断两者裁定结论是否一致。")
    res = judge_llm.chat([Message(role="system", content=sys_p),
                          Message(role="user", content=user)],
                         temperature=0.0, max_tokens=512, thinking=False)
    text = res.text.strip()
    first = text.splitlines()[0].strip().upper() if text else ""
    # 严格白名单：只认首行 YES/NO；其余（含"不一致"）一律视为 judge 异常
    if first.startswith("YES"):
        return True, text
    if first.startswith("NO"):
        return False, text
    return None, text


def check_citations(answer: str, must_cite: list[str]) -> dict:
    """核查 must_cite 是否出现在答案。卡 id 同时认 '卡牌#<id>' 与裸 '<id>'。"""
    hits = {}
    for c in must_cite:
        hit = (c in answer) or (f"#{c}" in answer) or (f"卡牌#{c}" in answer)
        hits[c] = hit
    return hits


def main():
    cases = load_cases()
    if not cases:
        print("cases.jsonl 为空，先写判例。"); return
    # num_few_shots=0 跑评测，无 fewshot 注入 → 全部判例可作评测题（无泄漏）
    excluded = set()  # 不排除：eval 用 0 few-shot
    evals = eval_cases(cases, excluded)
    print(f"[eval] 判例 {len(evals)} 条（num_few_shots=0，无泄漏）")

    store = CardStore(cache_dir=SETTINGS.cache_dir, base=SETTINGS.ygocdb_base)
    retriever = None
    try:
        from cli import _load_retriever
        retriever = _load_retriever(SETTINGS)
    except Exception as e:
        print(f"[eval] 无 RAG 索引({e})，rag_search 将返回空")
    agent = make_agent(store, retriever)
    judge_llm = LLMClient("openai", base_url=SETTINGS.llm_base_url,
                          api_key=SETTINGS.llm_api_key, model=SETTINGS.llm_model)

    rows = []
    n_correct, n_judged, n_judge_err, n_cite_total, n_cite_hit = 0, 0, 0, 0, 0
    for i, c in enumerate(evals, 1):
        q = c["q"]; expected = c.get("ruling", ""); must = c.get("must_cite", []) or []
        print(f"\n[{i}/{len(evals)}] {c['id']} {c['task']}: {q[:40]}")
        try:
            answer, trace = agent.ask(q, task=c.get("task", "ruling"),
                                       num_few_shots=0, k=8,
                                       max_context=SETTINGS.default_max_context_tokens,
                                       persona="strict")
        except Exception as e:
            answer, trace = f"(执行异常: {e})", None
        ok, reason = judge_correctness(judge_llm, q, expected, answer)
        cite = check_citations(answer, must)
        cite_hit = sum(1 for v in cite.values() if v)
        if ok is None:
            n_judge_err += 1
            verdict = "JUDGE_ERROR"
        else:
            n_judged += 1
            n_correct += 1 if ok else 0
            verdict = "YES" if ok else "NO"
        n_cite_total += len(cite); n_cite_hit += cite_hit
        print(f"  判定: {verdict} | 引用 {cite_hit}/{len(cite)} | tools={trace.tools if trace else []}")
        rows.append({"id": c["id"], "task": c.get("task", ""),
                     "question": q, "expected": expected, "answer": answer,
                     "correct": verdict, "judge_reason": reason,
                     "must_cite": ";".join(must),
                     "cited": ";".join(f"{k}:{'Y' if v else 'N'}" for k, v in cite.items())})

    acc = n_correct / n_judged if n_judged else 0
    cite_rate = n_cite_hit / n_cite_total if n_cite_total else 0
    print("\n=== 汇总 ===")
    print(f"判例数: {len(evals)}（可判定 {n_judged}，judge 异常 {n_judge_err}）")
    print(f"正确率: {n_correct}/{n_judged} = {acc:.1%}")
    print(f"引用率: {n_cite_hit}/{n_cite_total} = {cite_rate:.1%}")

    out = Path(__file__).parent / "result.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "task", "question", "expected",
                                          "answer", "correct", "judge_reason",
                                          "must_cite", "cited"])
        w.writeheader(); w.writerows(rows)
    print(f"详情: {out}")
    store.close()


if __name__ == "__main__":
    main()
