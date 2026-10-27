"""向量化（★ 学习点：embedding 选择如何影响检索质量）。

为不阻塞学习，默认用**零依赖的 TF-IDF（标准库）**，立即可跑；
后续可无缝切到：
  - OpenAI 兼容 embedding（设 EMBED_MODEL + LLM_API_KEY）
  - 本地 sentence-transformers（设 EMBED_LOCAL_MODEL，装 [embed] extras）

三种后端输出统一归一化向量（list[float]），对上层透明。
"""
from __future__ import annotations
import math
import re
from typing import Iterable

# 简易中英文分词：中文按字、英文按词。学习用，非最优。
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _TfidfBackend:
    """极简 TF-IDF：用 corpus 统计 IDF，向量为词频*idf，L2 归一化。"""

    def __init__(self):
        self.idf: dict[str, float] = {}
        self.vocab: dict[str, int] = {}

    def fit(self, docs: Iterable[str]) -> None:
        docs = list(docs)
        n = len(docs)
        df: dict[str, int] = {}
        for d in docs:
            seen = set(tokenize(d))
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        # 平滑 IDF，避免除零
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self._dim = max(len(self.vocab), 1)

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for t in tokenize(text):
            if t in self.idf:
                vec[self.vocab[t]] += self.idf[t]
        return _l2norm(vec)

    @property
    def dim(self) -> int:
        return self._dim


def _l2norm(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class Embedder:
    """统一 embedding 接口。

    backend:
      'tfidf'      —— 零依赖，本地即时（默认）
      'openai'     —— OpenAI 兼容 /embeddings（需 EMBED_MODEL + Key）
      'local'      —— sentence-transformers（需装 extras）
    """

    def __init__(self, backend: str = "tfidf", *, base_url: str = "",
                 api_key: str = "", model: str = "", local_model: str = "",
                 dim: int = 0):
        self.backend = backend
        self._tfidf = _TfidfBackend()
        self._openai = None
        self._local = None
        self._fitted = False
        self._dim_param = dim  # openai 后端：>0 时请求指定维度（embedding-3 支持）
        if backend == "openai":
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("装 [llm] extras: pip install -e .[llm]") from e
            self._openai = OpenAI(base_url=base_url, api_key=api_key)
            self._model = model or "embedding-3"
        elif backend == "local":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("装 [embed] extras: pip install -e .[embed]") from e
            self._local = SentenceTransformer(local_model)

    def fit(self, docs: Iterable[str]) -> None:
        """TF-IDF 必须先 fit；其它后端忽略。"""
        if self.backend == "tfidf":
            self._tfidf.fit(docs)
            self._fitted = True

    @property
    def dim(self) -> int:
        if self.backend == "tfidf":
            return self._tfidf.dim
        if self.backend == "local":
            return int(self._local.get_sentence_embedding_dimension())
        return 0  # openai 由 model 决定，延迟知道

    def _openai_kwargs(self) -> dict:
        return {"dimensions": self._dim_param} if self._dim_param else {}

    def embed(self, text: str) -> list[float]:
        if self.backend == "tfidf":
            if not self._fitted:
                # 未 fit 时退化为纯词频归一化（仅 query 自比可用）
                return _l2norm([0.0])
            return self._tfidf.embed(text)
        if self.backend == "openai":
            r = self._openai.embeddings.create(model=self._model, input=text, **self._openai_kwargs())
            return r.data[0].embedding
        if self.backend == "local":
            v = self._local.encode(text, normalize_embeddings=True)
            return v.tolist()
        raise ValueError(self.backend)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "tfidf":
            return [self.embed(t) for t in texts]
        if self.backend == "openai":
            r = self._openai.embeddings.create(model=self._model, input=texts, **self._openai_kwargs())
            # 按 index 排序保证顺序
            r.data.sort(key=lambda d: d.index)
            return [d.embedding for d in r.data]
        if self.backend == "local":
            vs = self._local.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vs]
        raise ValueError(self.backend)
