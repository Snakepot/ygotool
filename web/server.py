"""FastAPI Web 后端：复用 CardStore/Retriever/CardAgent，暴露 REST + SSE 流式聊天。

启动：python cli.py serve，或 uvicorn web.server:app --port 8000
"""
from __future__ import annotations
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import get_settings
from ygocdb.store import CardStore
from ygocdb.codec import decode_card
from rag.chunker import strip_html
from rag.embedder import Embedder
from rag.index import VectorIndex
from rag.retriever import Retriever
from agent.llm import LLMClient
from agent.budget import TokenBudgeter
from agent.agent import CardAgent
from agent.tools import ToolRuntime

STATIC_DIR = Path(__file__).parent / "static"

# ---- 单例（进程级）----
settings = get_settings()
store = CardStore(cache_dir=settings.cache_dir, base=settings.ygocdb_base)
# 索引按需加载
_retriever: Retriever | None = None


def _load_retriever() -> Retriever | None:
    global _retriever
    if _retriever is not None:
        return _retriever
    idx_path = settings.cache_dir / "idx_main.db"
    if not idx_path.exists():
        return None
    emb = Embedder(backend="tfidf")
    idx = VectorIndex(idx_path)
    emb.fit(idx.all_chunk_texts())
    _retriever = Retriever(emb, idx)
    return _retriever


def _make_llm() -> LLMClient:
    if settings.llm_api_key:
        return LLMClient("openai", base_url=settings.llm_base_url,
                          api_key=settings.llm_api_key, model=settings.llm_model)
    return LLMClient("mock")


def _make_agent() -> CardAgent:
    return CardAgent(store, _make_llm(), retriever=_load_retriever(),
                    budgeter=TokenBudgeter(max_context=settings.default_max_context_tokens,
                                           reserve_completion=settings.default_max_tokens),
                    settings=settings)


app = FastAPI(title="ygotool")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---- 请求模型 ----
class ChatRequest(BaseModel):
    question: str
    task: str = "ruling"
    temperature: float | None = None
    max_tokens: int | None = None
    num_few_shots: int = 2
    max_context: int | None = None
    thinking: bool | None = None
    persona: str = "strict"
    history: list[dict] | None = None  # [{role, content}] 多轮


# ---- 端点 ----
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    # 简单 SVG 卡牌图标，消掉 404
    from fastapi.responses import Response
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
           '<rect x="14" y="6" width="36" height="52" rx="5" fill="#2b6cb0"/>'
           '<text x="32" y="40" font-size="26" text-anchor="middle" '
           'fill="#fff" font-family="sans-serif">YG</text></svg>')
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/params")
def get_params():
    return {
        "temperature": settings.default_temperature,
        "max_tokens": settings.default_max_tokens,
        "max_context": settings.default_max_context_tokens,
        "model": settings.llm_model,
        "backend": "openai" if settings.llm_api_key else "mock",
        "has_index": (settings.cache_dir / "idx_main.db").exists(),
    }


@app.get("/api/search")
def api_search(q: str = Query(...), limit: int = Query(10, ge=1, le=50)):
    hits = store.search_by_name(q, limit=limit)
    out = []
    for h in hits:
        dec = decode_card(h["data"])
        out.append({
            "cid": h["cid"], "id": h["id"], "cn_name": h["cn_name"],
            "sc_name": h.get("sc_name"), "jp_name": h.get("jp_name"),
            "types": (h["text"]["types"].splitlines() or [""])[0],
            "attribute": dec["attribute"], "race": dec["race"], "type": dec["type"],
            "level": dec["level"], "atk": dec["atk"], "def": dec["def"],
        })
    return {"result": out}


@app.get("/api/card/{card_id}")
def api_card(card_id: int, faq: bool = False):
    c = store.get_by_id(card_id)
    if not c:
        raise HTTPException(404, f"未找到 id={card_id}")
    dec = decode_card(c["data"])
    text = c["text"]
    data = {
        "cid": c["cid"], "id": c["id"], "cn_name": c["cn_name"],
        "sc_name": c.get("sc_name"), "jp_name": c.get("jp_name"),
        "en_name": c.get("en_name"),
        "types": text["types"], "pdesc": text.get("pdesc", ""),
        "desc": text.get("desc", ""), "decoded": dec,
    }
    if faq:
        d = store.fetch_detail(card_id)
        faqs = (d.get("faqs") or []) if d else []
        data["faqcount"] = len(faqs)
        data["faqs"] = [
            {"q": strip_html(f.get("question", "")),
             "a": strip_html(f.get("answer", ""))}
            for f in faqs[:30]
        ]
    return data


@app.get("/api/effect-search")
def api_effect_search(q: str = Query(...)):
    """按效果体(非cost)全量搜，返回现代/未确认两档完整名单(供浏览面板分页)。"""
    from agent.tools import ToolRuntime
    rt = ToolRuntime(store)
    modern, uncertain = rt.search_by_effect(q)
    return {"modern": modern, "uncertain": uncertain,
            "count": len(modern) + len(uncertain)}


@app.get("/api/action-search")
def api_action_search(q: str = Query(...), reason: str = ""):
    """按裁定动作本体全量搜（§五 #2），支持 cost/effect 过滤。供浏览面板。
    reason=cost 只命中 cost 段；reason=effect 只命中效果段；空则都要。
    无 reason 且本体未命中→回退 search_by_effect（子串兜底，覆盖未映射措辞）。
    返回 mode 标明命中路径（action/effect），供前端显示。
    """
    from agent.tools import ToolRuntime
    rt = ToolRuntime(store)
    rf = reason if reason in ("cost", "effect") else None
    modern, uncertain = rt.search_by_action(q, reason=rf)
    mode = "action"
    if not (modern or uncertain) and not rf:
        modern, uncertain = rt.search_by_effect(q)
        mode = "effect"
    return {"modern": modern, "uncertain": uncertain, "mode": mode,
            "count": len(modern) + len(uncertain)}


@app.get("/api/series-search")
def api_series_search(q: str = Query(...)):
    """按卡名全量搜某系列，返回全部命中(供浏览面板)。"""
    from agent.tools import ToolRuntime
    rt = ToolRuntime(store)
    cards, token = rt.search_by_series(q)
    return {"token": token, "cards": cards, "count": len(cards)}


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    agent = _make_agent()
    import asyncio
    from agent.llm import Message
    history = [Message(role=h["role"], content=h["content"])
               for h in (req.history or [])] or None

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def run_sync():
            try:
                for ev in agent.ask_stream(
                    req.question, task=req.task,
                    num_few_shots=req.num_few_shots,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    max_context=req.max_context,
                    thinking=req.thinking,
                    persona=req.persona,
                    history=history,
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(ev), loop).result()
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "error", "message": str(e)}), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        # run_in_executor 返回 Future（非 coroutine），用 ensure_future 调度
        asyncio.ensure_future(loop.run_in_executor(None, run_sync))
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                # 心跳：推理模型思考很久时保活连接（SSE 注释，浏览器忽略）
                yield ": ping\n\n"
                continue
            if ev is None:
                break
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})
