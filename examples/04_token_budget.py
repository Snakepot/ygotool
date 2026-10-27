"""学习示例 04：Token 预算扫描（★ Token 控制学习点）。

对同一道裁定题，扫不同 max_context_tokens，观察：
  - 检索 chunk 被保留多少 / 丢了多少
  - 是否触发 few-shot 被裁
  - 最终 prompt token 是否在预算内

跑法：python examples/04_token_budget.py
后端：默认 mock（看预算如何砍检索）；配好 .env 后改 backend='openai' 看砍检索后回答是否仍准。
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from ygocdb.store import CardStore
from rag.embedder import Embedder
from rag.index import VectorIndex
from rag.retriever import Retriever, build_index
from agent.llm import LLMClient
from agent.budget import TokenBudgeter
from agent.agent import CardAgent

QUESTION = "异色眼灵摆龙被破坏时的效果"
BUDGETS = [200, 600, 1500, 4000]

# 建一个小型 effect 索引用作检索源（离线，从本地卡池）
SAMPLE_NAMES = ["青眼白龙", "黑魔术师", "异色眼灵摆龙", "异色眼龙",
                "真红眼黑星龙", "增援", "死者苏生", "神风防护"]


def build_retriever(settings, store):
    cards = []
    seen = set()
    for name in SAMPLE_NAMES:
        for h in store.search_by_name(name, limit=2):
            if h["cid"] not in seen:
                seen.add(h["cid"])
                cards.append(h)
    emb = Embedder(backend="tfidf")
    idx = VectorIndex(settings.cache_dir / "idx_budget_demo.db")
    idx.clear()
    build_index(cards, "effect", emb, idx, progress=False)
    return Retriever(emb, idx), emb, idx


def make_llm(settings):
    if settings.llm_api_key:
        return LLMClient("openai", base_url=settings.llm_base_url,
                          api_key=settings.llm_api_key, model=settings.llm_model)
    print("[note] 未配置 LLM_API_KEY，使用 mock 后端（看预算如何砍检索）")
    return LLMClient("mock")


def main() -> None:
    s = get_settings()
    store = CardStore(cache_dir=s.cache_dir, base=s.ygocdb_base)
    if not store.db_path.exists():
        store.refresh()
    retriever, emb, idx = build_retriever(s, store)
    llm = make_llm(s)

    print(f"问题: {QUESTION} | 后端: {llm.backend}\n")
    print(f"{'预算':<8}{'系统':<6}{'问题':<6}{'shot':<10}{'检索(留/总)':<14}{'检索tok':<9}{'合计':<8}{'状态':<8}回答")
    print("-" * 100)
    for b in BUDGETS:
        budgeter = TokenBudgeter(max_context=b, reserve_completion=64)
        agent = CardAgent(store, llm, retriever=retriever,
                          budgeter=budgeter, settings=s)
        ans, trace = agent.ask(QUESTION, task="ruling",
                                num_few_shots=2, k=8)
        r = trace.budget
        print(f"{b:<8}{r.tokens_system:<6}{r.tokens_query:<6}"
              f"{r.shots_kept}/{r.shots_total:<9}"
              f"{r.chunks_kept}/{r.chunks_total:<13}{r.tokens_retrieved:<9}"
              f"{r.tokens_total:<8}{'OK' if r.within_budget else 'OVER':<8}"
              f"{ans[:24].replace(chr(10),' ')}")

    # 清理
    store.close()
    idx.db.close()
    import os
    p = s.cache_dir / "idx_budget_demo.db"
    if p.exists():
        try:
            os.remove(p)
        except PermissionError:
            pass

    print("\n学习观察：")
    print(" - 预算越小→检索 chunk 丢得越多（先丢低分）；极小时连 few-shot 都被裁；")
    print(" - 实践：要在小窗口模型上跑，就得靠预算管理；窗口大时尽量多塞检索证据。")


if __name__ == "__main__":
    main()
