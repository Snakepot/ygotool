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
