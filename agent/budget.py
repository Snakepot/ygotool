"""Token 预算与上下文截断（★ 学习点：Token 控制）。

给定 max_context_tokens，对"系统提示 + few-shot + 检索 chunk + 当前问题"按优先级装填：
  优先级：系统 > 当前问题 > few-shot > 检索 chunk（按相似度从高到低）
超预算时，先丢低分检索 chunk，再丢 few-shot，绝不丢系统与问题。
预留 reserve_completion 给模型输出。

学习观察（examples/04）：扫不同预算，看检索被砍到几个、回答是否仍准确。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .llm import Message, _count_tokens
from .prompts import build_system, FEW_SHOTS, build_messages
from rag.chunker import Chunk


@dataclass
class BudgetReport:
    tokens_system: int
    tokens_query: int
    shots_kept: int
    shots_total: int
    chunks_kept: int
    chunks_total: int
    tokens_retrieved: int
    tokens_total: int
    max_context: int
    within_budget: bool

    def __str__(self) -> str:
        flag = "✓" if self.within_budget else "✗ 超预算"
        return (f"系统{self.tokens_system}+问题{self.tokens_query}+few-shot"
                f"{self.shots_kept}/{self.shots_total}+检索{self.chunks_kept}/"
                f"{self.chunks_total}({self.tokens_retrieved}tok)={self.tokens_total}/"
                f"{self.max_context} {flag}")


class TokenBudgeter:
    def __init__(self, max_context: int = 6000, reserve_completion: int = 1024,
                 max_chunks: int = 20, per_chunk_overhead: int = 8):
        self.max_context = max_context
        self.reserve_completion = reserve_completion
        self.max_chunks = max_chunks
        self.per_chunk_overhead = per_chunk_overhead  # 每条 chunk 的编号/分隔开销

    def count(self, text: str) -> int:
        return _count_tokens(text)

    def fit(self, task: str, query: str, num_few_shots: int,
            retrieved: list[tuple[Chunk, float]],
            persona: str = "strict",
            history: list[Message] | None = None
            ) -> tuple[list[Message], BudgetReport]:
        """装填上下文。retrieved 须已按 score 降序。返回 (messages, report)。

        优先级：system > 当前问题 > 历史(最新优先,旧的可裁) > few-shot > 检索chunk。
        """
        sys_text = build_system(task, persona)
        usable = self.max_context - self.reserve_completion
        t_sys = self.count(sys_text)
        t_q = self.count(query)
        remaining = usable - t_sys - t_q

        # 历史：最新优先保留，超预算则丢最旧
        kept_history: list[Message] = []
        t_history = 0
        if history:
            for m in reversed(history):
                t = self.count(m.content)
                if t <= remaining:
                    kept_history.append(m)
                    remaining -= t
                    t_history += t
                else:
                    break  # 更旧的更可能丢
            kept_history.reverse()
            kept_history = kept_history  # chronological

        # few-shot：按顺序贪心装
        shots_all = FEW_SHOTS.get(task, [])[:num_few_shots]
        shots_kept = 0
        for q, a in shots_all:
            t = self.count(q) + self.count(a)
            if t <= remaining:
                shots_kept += 1
                remaining -= t
            else:
                break  # 顺序很重要，断掉后续

        # 检索 chunk：按 score 降序贪心装（不 break，让小 chunk 仍有机会）
        kept: list[tuple[Chunk, float]] = []
        t_retrieved = 0
        for ch, score in retrieved:
            if len(kept) >= self.max_chunks:
                break
            t = self.count(ch.text) + self.per_chunk_overhead
            if t <= remaining:
                kept.append((ch, score))
                remaining -= t
                t_retrieved += t

        retrieved_text = self._render(kept)
        messages = build_messages(task, query, num_few_shots=shots_kept,
                                  retrieved=retrieved_text, persona=persona,
                                  history=kept_history or None)
        total = (t_sys + t_q + t_history
                 + sum(self.count(q) + self.count(a) for q, a in shots_all[:shots_kept])
                 + t_retrieved)
        return messages, BudgetReport(
            tokens_system=t_sys, tokens_query=t_q,
            shots_kept=shots_kept, shots_total=len(shots_all),
            chunks_kept=len(kept), chunks_total=len(retrieved),
            tokens_retrieved=t_retrieved, tokens_total=total,
            max_context=self.max_context, within_budget=total <= usable,
        )

    @staticmethod
    def _render(kept: list[tuple[Chunk, float]]) -> str:
        if not kept:
            return ""
        lines = []
        for i, (ch, score) in enumerate(kept, 1):
            src = ch.meta.get("name", "?")
            # 引用统一用密码 id（前端可点开详情），无 id 才退回 cid
            nid = ch.meta.get("id") or ch.meta.get("cid") or "?"
            lines.append(f"[{i}] (相关度{score:.2f}, 来源:{src}#{nid})\n{ch.text}")
        return "\n\n".join(lines)

    # ---- ReAct 路径：对消息序列做"按块裁剪"（P0-1）----
    def fit_messages(self, messages: list[Message], q_idx: int,
                     reserve_completion: int | None = None
                     ) -> tuple[list[Message], BudgetReport]:
        """对 ReAct 的多轮消息序列做预算裁剪，保证 tool_calls 与 tool 结果不被拆散。

        与 fit() 的区别：fit() 装填"检索上下文"；这里裁剪"消息序列"。
        消息序列结构：system + few-shot/历史 + 当前问题 + 若干轮 assistant(tool_calls)/tool。
        **不能按单条消息丢**——assistant 的 tool_calls 与其 tool 回复必须整体保留或整体丢弃，
        否则违反 function-calling 协议（P0-3 同源问题）。

        优先级（与 fit 一致）：system > 当前问题 > 最近轮次 > few-shot/历史。
        裁剪顺序：先丢最旧的工具轮（保留最近一轮），再丢 few-shot/历史；system 与问题绝不丢。
        q_idx：当前问题在 messages 中的下标；返回新的 q_idx'（问题位置可能因裁剪前移）。
        """
        usable = self.max_context - (reserve_completion if reserve_completion is not None
                                     else self.reserve_completion)

        def size(m: Message) -> int:
            return self.count(m.content or "") + 4  # +4 近似 role/分隔开销

        total = sum(size(m) for m in messages)
        q_idx = max(0, min(q_idx, len(messages) - 1))
        question = messages[q_idx] if messages else None
        tokens_system = size(messages[0]) if messages else 0
        tokens_query = size(question) if question is not None else 0

        def build_report(kept: list[Message], qpos: int) -> BudgetReport:
            kept_tools = [m for m in kept if m.role == "tool"]
            all_tools = [m for m in messages if m.role == "tool"]
            head = kept[1:qpos] if qpos > 0 else []
            return BudgetReport(
                tokens_system=tokens_system, tokens_query=tokens_query,
                shots_kept=sum(1 for m in head if m.role == "user"),
                shots_total=sum(1 for m in messages[1:q_idx] if m.role == "user"),
                chunks_kept=len(kept_tools), chunks_total=len(all_tools),
                tokens_retrieved=sum(size(m) for m in kept_tools),
                tokens_total=sum(size(m) for m in kept),
                max_context=self.max_context,
                within_budget=sum(size(m) for m in kept) <= usable,
            )

        if total <= usable:
            return messages, build_report(messages, q_idx)

        # 构建可丢弃单元（块）：head = system 与问题之间（few-shot/历史，成对）；tail = 问题之后（工具轮/通知）
        head_units: list[list[int]] = []
        i = 1
        while i < q_idx:
            if (i + 1 < q_idx and messages[i].role == "user"
                    and messages[i + 1].role == "assistant"
                    and not messages[i + 1].tool_calls):
                head_units.append([i, i + 1])
                i += 2
            else:
                head_units.append([i])
                i += 1

        tail_units: list[tuple[str, list[int]]] = []
        j = q_idx + 1
        while j < len(messages):
            if messages[j].role == "assistant" and messages[j].tool_calls:
                k = j + 1
                while k < len(messages) and messages[k].role == "tool":
                    k += 1
                tail_units.append(("tool", list(range(j, k))))  # 整个工具轮作为一个块
                j = k
            elif messages[j].role == "user":
                tail_units.append(("notice", [j]))              # 撞上限/收尾通知
                j += 1
            else:
                tail_units.append(("other", [j]))
                j += 1

        keep = [True] * len(messages)
        keep[q_idx] = True  # 保护当前问题

        def recalc() -> int:
            return sum(size(m) for idx, m in enumerate(messages) if keep[idx])

        cur = total
        # 1) 先丢"通知"（撞上限/收尾的 user 提示）：它们只是指令，丢了对协议无影响
        for kind, unit in tail_units:
            if cur <= usable:
                break
            if kind != "notice":
                continue
            for x in unit:
                keep[x] = False
            cur = recalc()
        # 2) 再丢最旧的工具轮（保留最近一轮）
        tool_units = [u for k, u in tail_units if k == "tool"]
        for unit in tool_units[:-1]:
            if cur <= usable:
                break
            for x in unit:
                keep[x] = False
            cur = recalc()
        # 3) 再丢 few-shot / 历史（从最旧开始）——优先于"最近一轮工具结果"
        for unit in head_units:
            if cur <= usable:
                break
            for x in unit:
                keep[x] = False
            cur = recalc()
        # 4) 仍超预算：连最近一轮工具结果也丢（保 system + 问题，硬保底）
        if cur > usable:
            for unit in tool_units[-1:]:
                for x in unit:
                    keep[x] = False
            cur = recalc()

        kept = [m for idx, m in enumerate(messages) if keep[idx]]
        qpos = kept.index(question) if question in kept else q_idx
        return kept, build_report(kept, qpos)
