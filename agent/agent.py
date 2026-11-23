"""Agent 编排（★ 学习点：工具调用编排 + 降幻觉）。

CardAgent 采用"工具增强"管线（非自由 ReAct，因 mock 后端不支持原生 tool-calling）：
  1. 按任务调对应工具拉真实数据（rag_search / search_card / get_limits）
  2. 工具结果作为检索 chunk 喂给 TokenBudgeter 装填
  3. LLM 综合→带来源引用的回答
  4. 返回回答 + trace（调了哪些工具、保留几个 chunk、token 用量、预算报告）

接真模型后，可在 step 1 处替换为原生 function-calling 的多轮 ReAct，trace 结构不变。
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator

from config import Settings
from ygocdb.store import CardStore
from rag.retriever import Retriever
from rag.chunker import Chunk, FaqChunker
from .llm import LLMClient, Message
from .budget import TokenBudgeter, BudgetReport
from .tools import ToolRuntime, TOOLS
from .prompts import build_system, FEW_SHOTS

# "按效果找卡"类问题意图：有哪些卡 / 什么卡 / 列出 / 拥有...的卡
_LIST_INTENT = re.compile(r"(有哪些|有什么|哪些卡|什么卡|列出|拥有|包含|带有).{0,20}(卡|效果)")
# "某系列所有卡"意图：系列 / 所有 / 全部 / 尽可能多 / 等卡
_SERIES_INTENT = re.compile(r"系列|所有|全部|尽可能多|等卡")
# 引号内容（中英文引号、『』「」''""），用三引号 raw 避免转义
_QUOTES = re.compile(r"""[「『“‟‘‛"']([^」』””’‛"']{2,})[」』””’‛"']""")
# 枚举样板词：从问句剥掉这些，剩下的就是动作/效果关键词（无引号时用）
# 注意：不剥裸"卡"——会误伤"卡组"；trailing 卡由下方 strip 收尾处理
_BOILER = re.compile(r"有哪些|有什么|哪些卡|什么卡|列出|拥有|包含|带有|的卡|的卡牌|卡牌|吗|呢|？\?|一些|些|的|了")


def _extract_list_keyword(question: str) -> str:
    """从枚举问句提取动作/效果关键词。优先引号内文本；无引号则剥样板词。"""
    qs = _QUOTES.findall(question)
    if qs:
        return qs[0]
    t = _BOILER.sub("", question or "").strip("的了么呢吗")
    return t


@dataclass
class AgentTrace:
    task: str
    tools: list[str] = field(default_factory=list)
    tool_results: dict = field(default_factory=dict)
    # P1-1：统一语义——实际进入最终上下文的检索/工具素材条数（预算裁剪后）。
    # mock 路径=prompt 中保留的 chunk 数；ReAct 路径=保留的 tool 结果条数。
    retrieved_total: int = 0
    budget: BudgetReport | None = None
    llm_usage: dict = field(default_factory=dict)
    answer: str = ""
    iterations: int = 0  # ReAct 循环轮数（mock 路径为 0）

    def __str__(self) -> str:
        lines = [f"[trace] task={self.task} tools={self.tools} iters={self.iterations}"]
        if self.budget:
            lines.append(f"[trace] budget: {self.budget}")
        lines.append(f"[trace] llm usage: {self.llm_usage}")
        return "\n".join(lines)

    def to_event(self) -> dict:
        """供流式端点发送的 trace 事件（首尾两处都用它，形状一致）。"""
        return {
            "type": "trace",
            "task": self.task,
            "tools": self.tools,
            "tool_results": self.tool_results,
            "retrieved_total": self.retrieved_total,
            "iterations": self.iterations,
            "llm_usage": self.llm_usage,
            "budget": asdict(self.budget) if self.budget else None,
        }


class CardAgent:
    def __init__(self, store: CardStore, llm: LLMClient,
                 retriever: Retriever | None = None,
                 budgeter: TokenBudgeter | None = None,
                 settings: Settings | None = None):
        self.store = store
        self.llm = llm
        self.retriever = retriever
        self.budgeter = budgeter or TokenBudgeter()
        self.settings = settings

    def _prepare(self, question: str, task: str, num_few_shots: int, k: int,
                 max_context: int | None, thinking: bool | None,
                 temperature: float | None, max_tokens: int | None,
                 persona: str = "strict",
                 history: list | None = None
                 ) -> tuple[list, AgentTrace, float, int, bool]:
        """工具调用 + 预算装填。返回 (messages, trace, temp, max_tokens, thinking)。"""
        rt = ToolRuntime(self.store, self.retriever, self.settings)
        trace = AgentTrace(task=task)
        retrieved: list[tuple[Chunk, float]] = []

        # "按效果找卡"意图：问题含引号且问"有哪些卡"→用 PSCT 解析按效果体搜全部
        # 枚举意图：不再要求引号——无引号时从问句剥样板词取关键词
        # （否则"有哪些舍弃手卡的卡"会退回 rag_search top-k，只看几张卡判"没有"，漏全量）
        list_intent = bool(_LIST_INTENT.search(question))
        list_answered = False
        if list_intent:
            from ygocdb.ontology import map_phrase_to_actions
            # cost/effect 意图从整句探测（"cost"常在引号外，仅靠引号内 kw 会漏）
            q = question or ""
            cost_intent = any(w in q for w in ("cost", "代价", "作为cost", "作为代价"))
            eff_intent = (any(w in q for w in ("效果处理", "作为效果", "效果时"))
                          and not cost_intent and "对方效果" not in q)
            reason_filter = "cost" if cost_intent else ("effect" if eff_intent else None)
            kws = _QUOTES.findall(question) or [_extract_list_keyword(question)]
            kws = [k for k in kws if len(k) >= 2]
            for kw in kws:
                # 命中已知动作→走结构化本体检索（能按 cost/effect 过滤）；否则回退子串检索
                use_action = bool(map_phrase_to_actions(kw))
                if use_action:
                    modern, uncertain = rt.search_by_action(kw, reason=reason_filter)
                    tool_name = "search_by_action"
                    tag = "action_search"
                    hdr = ("本体检索(按动作" +
                           (f"+仅{reason_filter}段" if reason_filter else "+cost/effect 都要") + ")")
                else:
                    modern, uncertain = rt.search_by_effect(kw)
                    tool_name = "search_by_effect"
                    tag = "effect_search"
                    hdr = "效果体含该动作,cost已剥离"
                if modern or uncertain:
                    list_answered = True
                    # 名单可能上百，整块塞会超预算被丢；给 LLM "总数+前N样例(带命中片段)"，
                    # 完整名单走前端浏览面板。带 snippet 让 LLM 有依据、不因严格 grounding 拒答。
                    SAMPLE = 15
                    lines = [f"共匹配 {len(modern)+len(uncertain)} 张（现代{len(modern)}+未确认{len(uncertain)}），"
                             f"{hdr}，下面列前 {SAMPLE} 样例（完整名单见左栏搜索）："]
                    if modern:
                        lines.append("【现代】:")
                        for c in modern[:SAMPLE]:
                            lines.append(f"- {c['cn_name']}(#{c['id']}) {c['types']} | "
                                         f"{c.get('snippet','')}")
                    if uncertain:
                        lines.append("【未确认(90X前旧措辞,cost/效果分不出,须查FAQ)】:")
                        for c in uncertain[:SAMPLE]:
                            lines.append(f"- {c['cn_name']}(#{c['id']}) {c['types']} | "
                                         f"{c.get('snippet','')}")
                    retrieved.append((Chunk(
                        text="\n".join(lines), meta={"kind": tag, "name": tag}), 1.8))
                    trace.tools.append(tool_name)
                    trace.tool_results.setdefault(tool_name, 0)
                    trace.tool_results[tool_name] += len(modern) + len(uncertain)

        # "某系列所有卡"意图：问题含 系列/所有/全部/尽可能多/等卡 → 按卡名全量搜
        series_intent = bool(_SERIES_INTENT.search(question))
        if series_intent:
            scards, token = rt.search_by_series(question)
            if scards:
                SAMPLE = 30
                lines = [f"系列「{token}」共 {len(scards)} 张，下面列前 {SAMPLE} 样例（完整名单见左栏 series 搜索）："]
                lines.append("、".join(f"{c['cn_name']}(#{c['id']})" for c in scards[:SAMPLE]))
                retrieved.append((Chunk(
                    text="\n".join(lines), meta={"kind": "series_search", "name": "series_search"}), 1.8))
                trace.tools.append("search_by_series")
                trace.tool_results["search_by_series"] = len(scards)

        if task == "ruling":
            linked = self._link_and_fetch(question, rt)
            retrieved.extend(linked)
            if linked:
                trace.tools.append("link_cards")
                trace.tool_results["link_cards"] = len(linked)
            # list_intent 已给出枚举名单时不跑 rag_search（名单即答案，避免噪音反噬）
            if not list_answered:
                kind = self._prefer_faq_kind()
                hits, _ = rt.rag_search(question, k=k, kind=kind)
                retrieved.extend(hits)
                if hits:
                    trace.tools.append("rag_search")

        elif task == "search":
            # 查卡：实体链接仅用本地数据（不联网拉 FAQ，避免限流）
            if not list_answered:
                linked = self._link_and_fetch(question, rt, max_cards=5, with_faq=False)
                retrieved.extend(linked)
                if linked:
                    trace.tools.append("link_cards")
                    trace.tool_results["link_cards"] = len(linked)
            if not list_answered:
                hits, _ = rt.rag_search(question, k=max(1, k - 1))
                if hits:
                    retrieved.extend(hits)
                    trace.tools.append("rag_search")

        elif task == "deck":
            if not list_answered:
                hits, _ = rt.rag_search(question, k=k)
                retrieved.extend(hits)
                if hits:
                    trace.tools.append("rag_search")
                seen: dict[int, tuple[str, str]] = {}
                for ch, _ in hits:
                    cid = ch.meta.get("id") or ch.meta.get("cid")
                    name = ch.meta.get("name", "?")
                    if cid and cid not in seen:
                        try:
                            seen[cid] = (name, rt.limit_status(cid, "cn"))
                        except Exception:
                            pass
                if seen:
                    trace.tools.append("limit_status")
                    trace.tool_results["limit_status"] = len(seen)
                    lines = [f"- {n}(id={cid}) 简中禁限：{st}"
                             for cid, (n, st) in seen.items()]
                    retrieved.append((Chunk(
                        text="检索到的候选卡及当前简中禁限状态：\n" + "\n".join(lines),
                        meta={"kind": "deck_candidates", "name": "candidates"}), 1.6))
            lim_txt, _ = rt.get_limits()
            trace.tools.append("get_limits")
            retrieved.append((Chunk(text=lim_txt, meta={"kind": "limits",
                                                        "name": "limits"}), 0.5))
        else:
            raise ValueError(f"未知任务: {task}")

        retrieved.sort(key=lambda x: x[1], reverse=True)

        budgeter = self.budgeter
        if max_context is not None:
            budgeter = TokenBudgeter(max_context=max_context,
                                     reserve_completion=self.budgeter.reserve_completion)
        messages, report = budgeter.fit(task, question, num_few_shots, retrieved, persona, history)
        trace.budget = report
        # P1-1：retrieved_total 统一为"进入最终上下文的素材条数"（预算裁剪后），
        # 与 ReAct 路径语义对齐；检索候选总数看 budget.chunks_total。
        trace.retrieved_total = report.chunks_kept

        temp = self.settings.default_temperature if (temperature is None and self.settings) else (temperature or 0.2)
        mt = max_tokens or (self.settings.default_max_tokens if self.settings else 1024)
        # 列举型查询（按效果找卡）不需要深思：search_by_text 已给候选，关 thinking 省 token
        th = thinking if thinking is not None else (task == "ruling" and not list_intent)
        return messages, trace, temp, mt, th

    def ask(self, question: str, task: str = "ruling",
            num_few_shots: int = 2, k: int = 8,
            temperature: float | None = None,
            max_tokens: int | None = None,
            max_context: int | None = None,
            thinking: bool | None = None,
            persona: str = "strict",
            history: list | None = None) -> tuple[str, AgentTrace]:
        # 真模型走 ReAct（LLM 自选工具）；mock 走旧确定性管线兜底
        if self.llm.backend != "mock":
            return self._react_ask(question, task, num_few_shots, k, max_context,
                                   thinking, temperature, max_tokens, persona, history)
        messages, trace, temp, mt, th = self._prepare(
            question, task, num_few_shots, k, max_context, thinking, temperature, max_tokens, persona, history)
        res = self.llm.chat(messages, temperature=temp, max_tokens=mt, thinking=th)
        trace.llm_usage = res.usage
        trace.answer = res.text
        return res.text, trace

    def ask_stream(self, question: str, task: str = "ruling",
                   num_few_shots: int = 2, k: int = 8,
                   temperature: float | None = None,
                   max_tokens: int | None = None,
                   max_context: int | None = None,
                   thinking: bool | None = None,
                   persona: str = "strict",
                   history: list | None = None) -> Iterator[dict]:
        """流式问答。先 yield trace 事件，再转发 LLM 的 reasoning/content/usage，最后 done。"""
        if self.llm.backend != "mock":
            yield from self._react_stream(question, task, num_few_shots, k, max_context,
                                           thinking, temperature, max_tokens, persona, history)
            return
        messages, trace, temp, mt, th = self._prepare(
            question, task, num_few_shots, k, max_context, thinking, temperature, max_tokens, persona, history)
        yield trace.to_event()
        for ev in self.llm.stream(messages, temperature=temp, max_tokens=mt, thinking=th):
            if ev.get("type") == "usage":
                trace.llm_usage = ev["usage"]
            yield ev
        yield {"type": "done"}

    # ---- ReAct：真模型原生 function-calling 多轮 ----
    MAX_ITERS = 4         # 最多 4 轮 LLM 调用（防跑飞）
    MAX_TOOL_CALLS = 10  # 单次问答累计工具调用上限（防逐卡循环）

    def _react_messages(self, question: str, task: str, num_few_shots: int,
                        persona: str, history: list | None) -> list[Message]:
        """构造 ReAct 初始消息：system(含工具规约) + few-shot + 历史 + 用户问。
        few-shot 优先取判例库（教 ReAct 格式），无则回退内置 FEW_SHOTS。"""
        from .data_loader import load_cases, fewshot_pairs
        msgs = [Message(role="system",
                        content=build_system(task, persona, react=True))]
        cases = load_cases()
        pairs = fewshot_pairs(cases, task, num_few_shots)
        if not pairs:
            pairs = [(q, a) for q, a in FEW_SHOTS.get(task, [])[:max(0, num_few_shots)]]
        for q, a in pairs:
            msgs.append(Message(role="user", content=q))
            msgs.append(Message(role="assistant", content=a))
        if history:
            msgs.extend(history)
        msgs.append(Message(role="user", content=question))
        return msgs

    def _react_params(self, task: str, thinking, temperature, max_tokens
                      ) -> tuple[float, int, bool]:
        temp = (temperature if temperature is not None
                else (self.settings.default_temperature if self.settings else 0.2))
        mt = max_tokens or (self.settings.default_max_tokens if self.settings else 1024)
        th = thinking if thinking is not None else (task == "ruling")
        return temp, mt, th

    def _react_ask(self, question, task, num_few_shots, k, max_context,
                   thinking, temperature, max_tokens, persona, history
                   ) -> tuple[str, AgentTrace]:
        rt = ToolRuntime(self.store, self.retriever, self.settings)
        trace = AgentTrace(task=task)
        msgs = self._react_messages(question, task, num_few_shots, persona, history)
        temp, mt, th = self._react_params(task, thinking, temperature, max_tokens)
        # 预算：默认用注入的 budgeter；调用方传了 max_context 时临时覆盖（与 _prepare 一致）
        budgeter = self.budgeter
        if max_context is not None:
            budgeter = TokenBudgeter(max_context=max_context,
                                     reserve_completion=self.budgeter.reserve_completion)
        q_idx = len(msgs) - 1              # 当前问题位置（预算裁剪时保护）
        question_msg = msgs[q_idx]
        usage = {"prompt": 0, "completion": 0, "total": 0}
        answer = ""
        total_calls = 0
        force_final = False
        # 第 1 轮只做工具路由（关 thinking 秒回）；拿到工具结果的后续轮次恢复深思（ruling），
        # 否则"模型自然收敛直接作答"的最终答会在 thinking 关闭状态下产出，与设计意图相悖。
        for it in range(self.MAX_ITERS):
            # P0-1：每轮调用前按预算裁剪消息序列（按协议块裁剪，tool_calls 与 tool 结果不拆散）
            msgs, report = budgeter.fit_messages(msgs, q_idx, reserve_completion=mt)
            trace.budget = report
            q_idx = next(i for i, m in enumerate(msgs) if m is question_msg)
            res = self.llm.chat(msgs, temperature=temp, max_tokens=mt,
                                thinking=th if (force_final or it > 0) else False,
                                tools=None if force_final else TOOLS)
            for key in usage:
                usage[key] += res.usage.get(key, 0)
            trace.iterations = it + 1
            if not res.tool_calls or force_final:
                answer = res.text or ""
                break
            # 模型要调工具：回灌 assistant(tool_calls) + 各 tool 结果
            msgs.append(Message(role="assistant", content=res.text or "", tool_calls=res.tool_calls))
            for tc in res.tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "")
                if total_calls >= self.MAX_TOOL_CALLS: # 先前达到上限会直接跳出，剩下的 tool_call 没有任何回复
                    # 因为function-calling 的对话格式是一个严格配对的序列，这会导致调用留在 assistant 消息里，却没有任何 tool 回复，协议违法
                    # OpenAI 协议要求每个 tool_call 都有对应 tool 消息；未执行的补占位，
                    # 否则消息序列非法（严格 API 会报协议错）。
                    msgs.append(Message(role="tool",
                                        content="(已达工具调用上限，该工具未执行)",
                                        tool_call_id=tc.get("id"), name=name))
                    continue
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                text, _tex = rt.call(name, args)
                trace.tools.append(name)
                trace.tool_results[name] = trace.tool_results.get(name, 0) + 1
                total_calls += 1
                msgs.append(Message(role="tool", content=text,
                                    tool_call_id=tc.get("id"), name=name))
            if total_calls >= self.MAX_TOOL_CALLS:
                force_final = True
                msgs.append(Message(role="user",
                                    content="(已达工具调用上限，请基于已获取信息直接给出最终回复，不要再调工具)"))
        else:
            answer = ""
        # 循环耗尽（模型一直调工具没收敛）或撞上限：补一次无 tools 调用逼出最终答
        if not answer:
            msgs.append(Message(role="user",
                                content="(已达到工具调用轮数上限，请基于已获取信息直接给出最终回复，不要再调工具)"))
            msgs, report = budgeter.fit_messages(msgs, q_idx, reserve_completion=mt)
            trace.budget = report
            res = self.llm.chat(msgs, temperature=temp, max_tokens=mt, thinking=th, tools=None)
            for key in usage:
                usage[key] += res.usage.get(key, 0)
            answer = res.text or "(工具调用达上限，未能完成推理)"
        trace.llm_usage = usage
        trace.answer = answer
        # P1-1：与 mock 路径统一——retrieved_total = 最终上下文中保留的素材条数
        trace.retrieved_total = trace.budget.chunks_kept if trace.budget else len(trace.tools)
        return answer, trace

    def _react_stream(self, question, task, num_few_shots, k, max_context,
                      thinking, temperature, max_tokens, persona, history
                      ) -> Iterator[dict]:
        rt = ToolRuntime(self.store, self.retriever, self.settings)
        trace = AgentTrace(task=task)
        msgs = self._react_messages(question, task, num_few_shots, persona, history)
        temp, mt, th = self._react_params(task, thinking, temperature, max_tokens)
        budgeter = self.budgeter
        if max_context is not None:
            budgeter = TokenBudgeter(max_context=max_context,
                                     reserve_completion=self.budgeter.reserve_completion)
        q_idx = len(msgs) - 1
        question_msg = msgs[q_idx]
        msgs, report = budgeter.fit_messages(msgs, q_idx, reserve_completion=mt)
        trace.budget = report
        q_idx = next(i for i, m in enumerate(msgs) if m is question_msg)
        yield trace.to_event()
        usage = {"prompt": 0, "completion": 0, "total": 0}
        total_calls = 0
        force_final = False
        got_answer = False  # 是否已流式吐出最终答（自然收敛 or force_final 轮）
        answer_parts: list[str] = []  # 累积已流出的正文，写回 trace.answer
        # 第 1 轮路由关 thinking 秒回；拿到工具结果的后续轮次开深思（ruling），
        # 否则自然收敛的最终答会在 thinking 关闭状态下产出。force_final 收尾轮同样开。
        for it in range(self.MAX_ITERS):
            tool_calls = None
            got_content = False  # 本轮的"内容产物"标记：只吐 reasoning 不算收敛；按轮重置，
                                 # 避免上一轮（带工具调用的轮）吐过的提示性 content 掩盖本轮空答
            # P0-1：每轮调用前按预算裁剪消息序列（按协议块裁剪，tool_calls 与 tool 结果不拆散）
            msgs, report = budgeter.fit_messages(msgs, q_idx, reserve_completion=mt)
            trace.budget = report
            q_idx = next(i for i, m in enumerate(msgs) if m is question_msg)
            for ev in self.llm.stream(msgs, temperature=temp, max_tokens=mt,
                                      thinking=th if (force_final or it > 0) else False,
                                      tools=None if force_final else TOOLS):
                t = ev.get("type")
                if t == "tool_calls":
                    tool_calls = ev["tool_calls"]
                elif t == "usage":
                    for key in usage:
                        usage[key] += ev["usage"].get(key, 0)
                    yield ev  # 与 mock 路径一致：usage 事件转发给前端
                elif t == "content":
                    got_content = True
                    answer_parts.append(ev.get("text", ""))
                    yield ev  # content 实时转发
                else:
                    yield ev  # reasoning / error 等
            trace.iterations = it + 1
            if not tool_calls or force_final:  # 先前是：本轮没有 tool_calls 就认为最终答案已经流出，会导致模型什么都没答
                if got_content:
                    got_answer = True
                    break  # 无工具调用且本轮吐过 content = 最终答
                break  # 只吐了 reasoning/空内容：不置 got_answer，交给下方兜底补齐(原本没有这个break会掉出循环，不触发兜底)
            msgs.append(Message(role="assistant", content="", tool_calls=tool_calls))
            for tc in tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "")
                if total_calls >= self.MAX_TOOL_CALLS:
                    # 协议要求：每个 tool_call 都要有对应 tool 消息；未执行的补占位
                    msgs.append(Message(role="tool",
                                        content="(已达工具调用上限，该工具未执行)",
                                        tool_call_id=tc.get("id"), name=name))
                    continue
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                text, _tex = rt.call(name, args)
                trace.tools.append(name)
                trace.tool_results[name] = trace.tool_results.get(name, 0) + 1
                total_calls += 1
                msgs.append(Message(role="tool", content=text,
                                    tool_call_id=tc.get("id"), name=name))
            if total_calls >= self.MAX_TOOL_CALLS:
                force_final = True
                msgs.append(Message(role="user",
                                    content="(已达工具调用上限，请基于已获取信息直接给出最终回复，不要再调工具)"))
        # 循环耗尽（模型一直调工具没收敛）或撞上限没机会收尾：补一次无 tools 流式逼出最终答
        if not got_answer:  # 任何"没成功产出正文"的退出都会走到这里
            msgs.append(Message(role="user",
                                content="(请基于已获取信息直接给出最终回复，不要再调工具)"))
            msgs, report = budgeter.fit_messages(msgs, q_idx, reserve_completion=mt)
            trace.budget = report
            for ev in self.llm.stream(msgs, temperature=temp, max_tokens=mt,
                                      thinking=th, tools=None):
                if ev.get("type") == "usage":
                    for key in usage:
                        usage[key] += ev["usage"].get(key, 0)
                    yield ev
                elif ev.get("type") == "content":
                    answer_parts.append(ev.get("text", ""))
                    yield ev
                else:
                    yield ev
        trace.llm_usage = usage
        trace.answer = "".join(answer_parts)
        # P1-1：与 mock 路径统一——retrieved_total = 最终上下文中保留的素材条数
        trace.retrieved_total = trace.budget.chunks_kept if trace.budget else len(trace.tools)
        # 最终 trace 与首帧同形（task/tools/budget/iterations/llm_usage 全量），
        # 前端无需区分 mock/ReAct 两条路径的字段差异。
        yield trace.to_event()
        yield {"type": "done"}

    def _link_and_fetch(self, question: str, rt: ToolRuntime,
                        max_cards: int = 2, with_faq: bool = True
                        ) -> list[tuple[Chunk, float]]:
        """实体链接：识别问题里的卡名，取其效果（本地）+ 可选 FAQ（在线）。

        with_faq=True（ruling 用）：在线拉 show=all 取官方 FAQ；超时/失败自动回退本地效果。
        with_faq=False（search 用）：只用本地 cards.db 的效果文本，不联网——快且不触发限流。
        返回 (chunk, score) 列表（高权重，预算裁剪时优先保留）。
        """
        out: list[tuple[Chunk, float]] = []
        linked = self.store.find_cards_in_text(question, limit=max_cards)
        from ygocdb.effect_parser import tag_actions, render_action_tags
        for card in linked:
            cid = card.get("cid")
            name = card.get("cn_name") or "?"
            text = card.get("text", {}) or {}  # 本地 cards.db 已有 text
            header = f"{name}（{text.get('types','').splitlines()[0]}）"
            effect = (f"{header}\n【效果】{text.get('desc','')}"
                      + (f"\n【灵摆效果】{text.get('pdesc','')}"
                         if text.get("pdesc") else ""))
            # §五 #2：附动作标签（cost/effect 区分），直接服务"cost≠效果"裁定
            tags = tag_actions(text.get("desc"), text.get("pdesc"))
            tag_line = render_action_tags(tags)
            if tag_line:
                effect += f"\n{tag_line}"
            out.append((Chunk(text=effect, meta={"kind": "linked", "cid": cid,
                                                  "id": card.get("id"), "name": name}), 2.0))
            if not with_faq:
                continue
            try:
                detail = self.store.fetch_detail(card["id"])
            except Exception:
                detail = None  # 联网失败回退：仅用本地效果，不阻断
            if not detail:
                continue
            for fc in FaqChunker().chunk(detail):
                fc.meta["kind"] = "linked_faq"
                out.append((fc, 1.9))
        return out

    def _prefer_faq_kind(self) -> str | None:
        """若向量库里有 faq chunk，裁定优先查 faq。"""
        if self.retriever is None:
            return None
        idx = self.retriever.index
        if idx.count("faq") > 0:
            return "faq"
        return None
