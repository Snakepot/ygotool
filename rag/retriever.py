"""检索器 + 建库便利函数。

Retriever：把 Embedder + VectorIndex 组合，提供 retrieve(query, k)。
build_index：批量把一批卡用指定切分器切分、向量化、入库。
"""
from __future__ import annotations
from typing import Iterable

from config import Settings
from .chunker import Chunk, chunk_cards
from .embedder import Embedder
from .embedder_factory import build_embedder_for_index
from .index import VectorIndex


class Retriever:
    def __init__(self, embedder: Embedder, index: VectorIndex):
        self.embedder = embedder
        self.index = index

    def retrieve(self, query: str, k: int = 5, kind: str | None = None,
                 filter_cid: int | None = None) -> list[tuple[Chunk, float]]:
        qv = self.embedder.embed(query)
        return self.index.search(qv, k=k, kind=kind, filter_cid=filter_cid)


def build_index(cards: Iterable[dict], chunker_name: str,
                embedder: Embedder, index: VectorIndex,
                batch: int | None = None, progress: bool = True,
                resume: bool = True) -> int:
    """对一批卡切分→向量化→入库。返回入库 chunk 数。

    TF-IDF 后端需先 fit，故两遍遍历：先收集所有 chunk 文本 fit，再批量 embed。
    resume=True 时跳过该 kind 下已入库 cid 的卡（断点续传）。
    openai 后端默认小批次(64)防超限；TF-IDF 大批次(500)。
    """
    cards = list(cards)
    if resume:
        done = index.indexed_cids(chunker_name)
        before = len(cards)
        cards = [c for c in cards if c.get("cid") not in done]
        if progress and len(cards) < before:
            print(f"[rag] resume: 跳过 {before - len(cards)} 张已入库卡，剩 {len(cards)} 张")
    chunks = chunk_cards(cards, chunker_name)
    if not chunks:
        return 0
    texts = [c.text for c in chunks]

    if embedder.backend == "tfidf":
        embedder.fit(texts)

    if batch is None:
        batch = 64 if embedder.backend == "openai" else 500

    total = 0
    for i in range(0, len(texts), batch):
        sub_t = texts[i:i + batch]
        sub_c = chunks[i:i + batch]
        vecs = embedder.embed_batch(sub_t)
        total += index.add(sub_c, vecs)
        if progress:
            print(f"[rag] {chunker_name}: 已入库 {total}/{len(chunks)}")
    return total


def build_faq_index(cards: Iterable[dict], store, chunker_name: str,
                    embedder: Embedder, index: VectorIndex,
                    batch: int | None = None, progress: bool = True,
                    resume: bool = True) -> int:
    """从在线 show=all 拉 FAQ 切分入库（裁定问答主力源）。

    cards: 各卡 dict（须含 cid 与 id），来自 cards.db。
    store: ygocdb.store.CardStore（用其 fetch_detail 的磁盘缓存，避免重复请求）。
    resume=True 时按 cid 跳过该 kind 下已入库卡（断点续传，省请求+省 embedding）。
    """
    from .chunker import get_chunker
    chunker = get_chunker(chunker_name)
    cards = list(cards)
    if resume:
        done = index.indexed_cids(chunker_name)
        before = len(cards)
        cards = [c for c in cards if c.get("cid") not in done]
        if progress and len(cards) < before:
            print(f"[rag-faq] resume: 跳过 {before - len(cards)} 张已入库卡，剩 {len(cards)} 张")
    if batch is None:
        batch = 64 if embedder.backend == "openai" else 200

    all_chunks: list = []
    for i, c in enumerate(cards, 1):
        cid = c.get("cid")
        card_id = c.get("id")
        if not card_id:
            continue
        try:
            detail = store.fetch_detail(card_id)
        except Exception as e:
            if progress:
                print(f"[rag-faq] id={card_id} 拉取失败({e})，跳过")
            continue
        if not detail:
            continue
        all_chunks.extend(chunker.chunk(detail))
        if progress and i % 25 == 0:
            print(f"[rag-faq] 进度 {i}/{len(cards)} 卡，已收集 {len(all_chunks)} chunk")
    if not all_chunks:
        return 0
    texts = [c.text for c in all_chunks]
    if embedder.backend == "tfidf":
        embedder.fit(texts)
    total = 0
    for i in range(0, len(texts), batch):
        vecs = embedder.embed_batch(texts[i:i + batch])
        total += index.add(all_chunks[i:i + batch], vecs)
    if progress:
        print(f"[rag-faq] {chunker_name}: 入库 {total} 个 chunk")
    return total


def load_retriever(settings: Settings, idx_name: str = "idx_main.db"
                   ) -> "Retriever | None":
    """加载已建索引，按 index_meta 记录的后端构造匹配 embedder（cli / web 共用）。

    ★ 踩坑（已修）：此前 web/server.py 各自实现成固定 TF-IDF，而索引是按
    embedding-3(1024 维) 建的 → 查询向量维度与库内不符，点积截断成垃圾分数。
    统一走本函数，保证"建库用的后端"与"查询用的后端"一致。
    TF-IDF 后端需从 chunk 文本重新 fit 词表（内存态）；openai/local 无状态直接用。
    meta 缺失（旧库）退化为 tfidf，保持向后兼容。
    """
    path = settings.cache_dir / idx_name
    if not path.exists():
        return None
    idx = VectorIndex(path)
    meta = idx.meta_all()
    emb = build_embedder_for_index(settings, meta)
    if emb.backend == "tfidf":
        emb.fit(idx.all_chunk_texts())
    return Retriever(emb, idx)
