"""ygotool 命令行入口。

子命令：
  refresh          按需同步本地卡池（MD5 增量）
  search <词>      本地按名称查卡（sqlite，离线）
  card <id>        按密码取卡详情 + 字段解码（可选 FAQ）
  limits           在线禁限卡表概览
  build-index      建 RAG 索引（默认 effect 切分；--all 全量、--resume 断点续传）
  build-faq-index  建 FAQ 跨卡索引（faqcount>0 的卡，裁定问答主力源，--resume 可续跑）
  agent <问题>     工具增强问答（需先 build-index；LLM 默认 mock，配 .env 接真模型）

示例：
  ygotool refresh
  ygotool search 青眼
  ygotool card 89631139
  ygotool build-index --n 500
  ygotool agent "异色眼灵摆龙被破坏时的效果" --task ruling
"""
from __future__ import annotations
import argparse
import sys

from config import get_settings
from ygocdb.store import CardStore
from ygocdb.codec import decode_card
from rag.index import VectorIndex
from rag.retriever import Retriever, build_index, build_faq_index
from rag.embedder_factory import build_embedder, build_embedder_for_index
from agent.llm import LLMClient
from agent.budget import TokenBudgeter
from agent.agent import CardAgent


def _store(settings) -> CardStore:
    return CardStore(cache_dir=settings.cache_dir, base=settings.ygocdb_base)


def _load_retriever(settings, idx_name="idx_main.db") -> Retriever | None:
    """加载已建索引，按 index_meta 记录的后端构造匹配 embedder。

    TF-IDF 后端需从 chunk 文本重新 fit 词表（内存态）；openai/local 无状态直接用。
    meta 缺失（旧库）退化为 tfidf，保持向后兼容。
    """
    path = settings.cache_dir / idx_name
    if not path.exists():
        return None
    idx = VectorIndex(path)
    meta = idx.meta_all()
    backend = meta.get("backend") or "tfidf"
    emb = build_embedder_for_index(settings, meta)
    if backend == "tfidf":
        emb.fit(idx.all_chunk_texts())
    return Retriever(emb, idx)


def cmd_refresh(args, settings):
    s = _store(settings)
    s.refresh()
    s.close()


def cmd_search(args, settings):
    s = _store(settings)
    hits = s.search_by_name(args.query, limit=args.limit)
    if not hits:
        print("未找到匹配卡牌。")
    for h in hits:
        dec = decode_card(h["data"])
        types = h["text"]["types"].splitlines()[0]
        print(f"{h['cn_name']:<14} id={h['id']:<10} {types} | "
              f"{dec['attribute']}/{dec['race']} {dec['level']} ATK{dec['atk']} DEF{dec['def']}")
    s.close()


def cmd_card(args, settings):
    s = _store(settings)
    c = s.get_by_id(int(args.id))
    if not c:
        print(f"未找到 id={args.id}（可先用 search 查）")
        s.close()
        return
    dec = decode_card(c["data"])
    print(f"【{c['cn_name']}】 id={c['id']} cid={c['cid']}")
    print(f"  {c['text']['types']}")
    print(f"  解码: 属性{dec['attribute']} 种族{dec['race']} 类型{dec['type']} "
          f"{dec['level']} ATK{dec['atk']} DEF{dec['def']} setcode{dec['setcode']}")
    print(f"  效果: {c['text']['desc']}")
    if c["text"].get("pdesc"):
        print(f"  灵摆: {c['text']['pdesc']}")
    if args.faq:
        d = s.fetch_detail(c["id"])
        faqs = d.get("faqs") or [] if d else []
        print(f"  官方FAQ: {len(faqs)} 条")
    s.close()


def cmd_limits(args, settings):
    s = _store(settings)
    lim = s.client.get_limits()
    for lang in ("cn", "ja", "en"):
        d = lim.get(lang, {})
        print(f"{lang}: 禁{len(d.get('forbidden', d.get('banned', [])))} "
              f"限{len(d.get('limited', []))} 准限{len(d.get('semiLimited', []))}")
    s.close()


def cmd_build_index(args, settings):
    s = _store(settings)
    if not s.db_path.exists():
        print("本地卡池不存在，先 refresh …")
        s.refresh()
    db = s.connect()
    if args.all:
        rows = db.execute("SELECT cid FROM cards").fetchall()
        scope = "全量"
    else:
        rows = db.execute("SELECT cid FROM cards LIMIT ?", (args.n,)).fetchall()
        scope = f"前 {args.n} 张（演示）"
    cards = [s.get_by_cid(r["cid"]) for r in rows if r["cid"] is not None]
    s.close()
    emb = build_embedder(settings)
    idx = VectorIndex(settings.cache_dir / "idx_main.db")
    if not args.resume:
        idx.clear(args.chunker)
    # 记录建库元数据，供加载时构造匹配 embedder
    idx.set_meta("backend", emb.backend)
    idx.set_meta("chunker", args.chunker)
    idx.set_meta("dim", settings.embed_dim or emb.dim or 0)
    if emb.backend == "openai":
        idx.set_meta("model", settings.embed_model or "embedding-3")
    n = build_index(cards, args.chunker, emb, idx, resume=args.resume)
    print(f"已建索引: {n} 个新 chunk（{args.chunker} 切分, {scope}, backend={emb.backend}）"
          f"→ cache/idx_main.db")


def cmd_build_faq_index(args, settings):
    s = _store(settings)
    if not s.db_path.exists():
        print("本地卡池不存在，先 ygotool refresh")
        s.close()
        return
    db = s.connect()
    # 注：cards.zip 不带 faqcount 字段（全 0），无法据此过滤"有 FAQ 的卡"；
    # FAQ 只能在线 show=all 逐卡取。故按 --n/--all/--ids 取卡，可中断续跑。
    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
        rows = db.execute(
            "SELECT cid, id FROM cards WHERE id IN (%s)" % ",".join("?" * len(ids)),
            ids,
        ).fetchall()
    elif args.all:
        rows = db.execute("SELECT cid, id FROM cards ORDER BY id").fetchall()
    else:
        rows = db.execute(
            "SELECT cid, id FROM cards ORDER BY id LIMIT ?", (args.n,)
        ).fetchall()
    cards = [{"cid": r["cid"], "id": r["id"]} for r in rows if r["id"]]
    s.close()
    scope = ("指定 %d 张" % len(cards)) if args.ids else ("全量 %d 张" % len(cards) if args.all else "前 %d 张(采样)" % len(cards))
    print(f"[build-faq-index] {scope}（在线拉 show=all，0.3s/卡限流，可中断续跑；无 FAQ 的卡返回空自动跳过）")
    emb = build_embedder(settings)
    idx = VectorIndex(settings.cache_dir / "idx_main.db")
    if not args.resume:
        idx.clear("faq")
    idx.set_meta("faq_backend", emb.backend)
    idx.set_meta("faq_model", settings.embed_model or "embedding-3")
    n = build_faq_index(cards, s, "faq", emb, idx, resume=args.resume)
    print(f"已建 FAQ 索引: {n} 个新 chunk（backend={emb.backend}）→ cache/idx_main.db")


def cmd_agent(args, settings):
    s = _store(settings)
    if not s.db_path.exists():
        print("本地卡池不存在，先 ygotool refresh")
        s.close()
        return
    ret = _load_retriever(settings)
    if ret is None:
        print("未找到 RAG 索引，先 ygotool build-index")
        s.close()
        return
    if settings.llm_api_key:
        llm = LLMClient("openai", base_url=settings.llm_base_url,
                         api_key=settings.llm_api_key, model=settings.llm_model)
    else:
        print("[note] 未配置 LLM_API_KEY，使用 mock 后端（接真模型请在 .env 配置）")
        llm = LLMClient("mock")
    budgeter = TokenBudgeter(max_context=settings.default_max_context_tokens,
                             reserve_completion=settings.default_max_tokens)
    agent = CardAgent(s, llm, retriever=ret, budgeter=budgeter, settings=settings)
    answer, trace = agent.ask(args.question, task=args.task,
                              num_few_shots=args.few_shots, k=args.k,
                              persona=args.persona)
    print(trace)
    print("\n— 回答 —")
    print(answer)
    s.close()


def cmd_serve(args, settings):
    try:
        import uvicorn
    except ImportError:
        print("缺少 web 依赖，先装：pip install -e \".[web]\"")
        return 1
    print(f"[serve] http://{args.host}:{args.port}  （Ctrl+C 退出）")
    if not (settings.cache_dir / "idx_main.db").exists():
        print("[serve] 提示：未找到 RAG 索引，建议先 ygotool build-index（裁定/构筑任务会弱）")
    uvicorn.run("web.server:app", host=args.host, port=args.port, log_level="info")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ygotool", description="ygocdb 本地卡牌工具 + Agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("refresh", help="同步本地卡池")
    a = sub.add_parser("search", help="按名称查卡"); a.add_argument("query"); a.add_argument("--limit", type=int, default=10)
    a = sub.add_parser("card", help="按密码取卡详情"); a.add_argument("id"); a.add_argument("--faq", action="store_true")
    sub.add_parser("limits", help="禁限卡表概览")
    a = sub.add_parser("build-index", help="建 RAG 索引（卡牌效果切分）")
    a.add_argument("--chunker", default="effect", choices=["card", "effect", "faq"])
    a.add_argument("--n", type=int, default=500, help="演示用取前 N 张（--all 时忽略）")
    a.add_argument("--all", action="store_true", help="全量卡池建索引（修召回弱）")
    a.add_argument("--resume", action="store_true", help="断点续传：跳过已入库 cid 的卡")
    a = sub.add_parser("build-faq-index", help="建 FAQ 跨卡索引（裁定主力源，在线逐卡取 FAQ）")
    a.add_argument("--n", type=int, default=200, help="采样取前 N 张卡（--all 时忽略）")
    a.add_argument("--all", action="store_true", help="全量卡池逐卡取 FAQ（很慢，可中断续跑）")
    a.add_argument("--ids", type=str, default="", help="指定 id 列表，逗号分隔（如 89631139,40640057）")
    a.add_argument("--resume", action="store_true", help="断点续传：跳过已入库 cid 的卡")
    a = sub.add_parser("agent", help="问答"); a.add_argument("question"); a.add_argument("--task", default="ruling", choices=["search", "ruling", "deck"]); a.add_argument("--k", type=int, default=8); a.add_argument("--few-shots", type=int, default=2); a.add_argument("--persona", default="strict", choices=["strict", "teach"])
    a = sub.add_parser("serve", help="启动 Web 界面"); a.add_argument("--host", default="127.0.0.1"); a.add_argument("--port", type=int, default=8000)

    args = p.parse_args(argv)
    settings = get_settings()
    {
        "refresh": cmd_refresh, "search": cmd_search, "card": cmd_card,
        "limits": cmd_limits, "build-index": cmd_build_index,
        "build-faq-index": cmd_build_faq_index, "agent": cmd_agent,
        "serve": cmd_serve,
    }[args.cmd](args, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
