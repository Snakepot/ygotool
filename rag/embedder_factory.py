"""Embedder 工厂（★ 学习点：后端可切、对上层透明）。

按 settings.embed_backend 构造 Embedder：
  - tfidf  零依赖兜底（须 fit）
  - openai 智谱 embedding-3（base_url/Key 留空回退 LLM 同一套）
  - local  sentence-transformers（bge）

cli 建库与 _load_retriever 加载都走这里，保证"建库用的后端"与"查询用的后端"一致。
"""
from __future__ import annotations
from config import Settings
from .embedder import Embedder


def build_embedder(settings: Settings) -> Embedder:
    be = settings.embed_backend
    if be == "tfidf":
        return Embedder("tfidf")
    if be == "openai":
        base_url = settings.embed_base_url or settings.llm_base_url
        api_key = settings.embed_api_key or settings.llm_api_key
        model = settings.embed_model or "embedding-3"
        return Embedder("openai", base_url=base_url, api_key=api_key,
                        model=model, dim=settings.embed_dim)
    if be == "local":
        return Embedder("local", local_model=settings.embed_local_model)
    raise ValueError(f"未知 embed_backend: {be}（可选: tfidf/openai/local）")


def build_embedder_for_index(settings: Settings, meta: dict) -> Embedder:
    """加载已建索引时用：按 index_meta 记录的 backend/model/dim 构造匹配 embedder，
    避免切后端后维度/模型错配（meta 由建库时写入）。"""
    be = meta.get("backend") or "tfidf"
    if be == "tfidf":
        return Embedder("tfidf")
    if be == "openai":
        base_url = settings.embed_base_url or settings.llm_base_url
        api_key = settings.embed_api_key or settings.llm_api_key
        model = meta.get("model") or settings.embed_model or "embedding-3"
        dim = int(meta.get("dim") or 0)
        return Embedder("openai", base_url=base_url, api_key=api_key,
                        model=model, dim=dim)
    if be == "local":
        return Embedder("local", local_model=settings.embed_local_model)
    raise ValueError(f"未知 embed_backend（来自 index_meta）: {be}")
