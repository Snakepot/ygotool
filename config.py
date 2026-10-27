"""配置加载：从 .env（若存在）与 os.environ 读取，提供统一访问。

为减少依赖，.env 解析自行实现，不依赖 python-dotenv。
阶段 1/2 不需要任何 LLM 配置即可运行。
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: str | Path = ".env") -> None:
    """极简 .env 加载器：只做 KEY=VALUE 注入到 os.environ（不覆盖已存在的环境变量）。"""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


@dataclass
class Settings:
    ygocdb_base: str = "https://ygocdb.com"
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))

    # LLM
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"

    # Embedding（★ 可切后端：tfidf 零依赖兜底 / openai 智谱 embedding-3 / local bge）
    embed_backend: str = "tfidf"
    embed_base_url: str = ""        # 空→回退 llm_base_url（智谱同一 base 即可调 /embeddings）
    embed_api_key: str = ""        # 空→回退 llm_api_key
    embed_model: str = "embedding-3"  # 智谱默认；TF-IDF/local 忽略
    embed_local_model: str = "BAAI/bge-small-zh-v1.5"
    embed_dim: int = 0             # >0 时请求指定维度（embedding-3 支持，省存储）；0 用模型默认

    # 默认推理参数（学习用：可被调用方覆盖以做对比实验）
    default_temperature: float = 0.2
    default_max_tokens: int = 1024
    default_max_context_tokens: int = 6000

    @classmethod
    def load(cls) -> "Settings":
        _load_dotenv()
        s = cls(
            ygocdb_base=os.environ.get("YGOCDB_BASE", cls.ygocdb_base),
            cache_dir=Path(os.environ.get("CACHE_DIR", "./cache")),
            llm_base_url=os.environ.get("LLM_BASE_URL", cls.llm_base_url),
            llm_api_key=os.environ.get("LLM_API_KEY", ""),
            llm_model=os.environ.get("LLM_MODEL", cls.llm_model),
            embed_backend=os.environ.get("EMBED_BACKEND", cls.embed_backend),
            embed_base_url=os.environ.get("EMBED_BASE_URL", cls.embed_base_url),
            embed_api_key=os.environ.get("EMBED_API_KEY", cls.embed_api_key),
            embed_model=os.environ.get("EMBED_MODEL", cls.embed_model),
            embed_local_model=os.environ.get("EMBED_LOCAL_MODEL", cls.embed_local_model),
            embed_dim=int(os.environ.get("EMBED_DIM", cls.embed_dim)),
            default_temperature=float(os.environ.get("DEFAULT_TEMPERATURE", cls.default_temperature)),
            default_max_tokens=int(os.environ.get("DEFAULT_MAX_TOKENS", cls.default_max_tokens)),
            default_max_context_tokens=int(os.environ.get("DEFAULT_MAX_CONTEXT_TOKENS", cls.default_max_context_tokens)),
        )
        s.cache_dir.mkdir(parents=True, exist_ok=True)
        return s


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings
