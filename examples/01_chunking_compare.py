"""学习示例 01：切分方式如何影响 RAG 召回（★ RAG 切分）。

对同一批卡，分别用 card / effect / faq 三种切分器建库，
对同一组问题检索 top-k，对比命中 chunk 的边界与重叠度。

跑法：python examples/01_chunking_compare.py
（默认零依赖 TF-IDF，无需 LLM Key）
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

# 一组与裁定/检索相关的问题，用于对比召回
QUESTIONS = [
    "被破坏时可以发动效果",
    "青眼白龙",
    "从卡组加入手卡",
    "无效化对方的攻击",
]

# 用于建库的卡名样本（学习演示用一小批，避免全量耗时）
SAMPLE_NAMES = [
    "青眼白龙", "黑魔术师", "死者苏生", "神圣魔术师", "暗黑骑士 盖亚",
    "异色眼灵摆龙", "真红眼黑星龙", "增援", "强制脱出装置", "神风防护",
]


def load_sample_cards(store: CardStore) -> list[dict]:
    out: list[dict] = []
    seen = set()
    for name in SAMPLE_NAMES:
        hits = store.search_by_name(name, limit=2)
        for h in hits:
            if h["cid"] in seen:
                continue
            seen.add(h["cid"])
            out.append(h)
    return out


def build_one(chunker_name: str, cards: list[dict], cache_dir: Path) -> Retriever:
    emb = Embedder(backend="tfidf")
    idx_path = cache_dir / f"idx_{chunker_name}.db"
    if idx_path.exists():
        idx_path.unlink()
    idx = VectorIndex(idx_path)
    n = build_index(cards, chunker_name, emb, idx, progress=False)
    print(f"  [{chunker_name}] 入库 {n} 个 chunk, dim={emb.dim}")
    return Retriever(emb, idx)


def main() -> None:
    s = get_settings()
    store = CardStore(cache_dir=s.cache_dir, base=s.ygocdb_base)
    if not store.db_path.exists():
        print("本地卡池不存在，先运行 refresh …")
        store.refresh()
    cards = load_sample_cards(store)
    print(f"样本卡 {len(cards)} 张\n")

    retrievers = {name: build_one(name, cards, s.cache_dir)
                  for name in ("card", "effect")}

    for q in QUESTIONS:
        print(f"\n=== 问题：{q} ===")
        for name, ret in retrievers.items():
            hits = ret.retrieve(q, k=3)
            print(f"  [{name}] top-3:")
            for ch, score in hits:
                snippet = ch.text[:50].replace("\n", " ")
                print(f"    {score:.3f}  {snippet}")

    print("\n学习观察：")
    print(" - card 切分召回到整张卡，上下文全但可能含无关效果；")
    print(" - effect 切分召回到单条效果，定位更细但可能丢失同卡其它相关条目；")
    print(" - faq 切分（需在线拉 show=all，阶段4接入后对比）最贴近裁定问答。")

    store.close()


if __name__ == "__main__":
    main()
