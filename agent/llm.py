"""LLM 客户端封装（★ 学习点：Temperature / max_tokens / Token 用量）。

统一接口 chat(...)，关键点：
  - temperature、max_tokens、model 均为显式参数，便于做扫描对比实验；
  - 每次调用记录 usage（prompt/completion/total tokens）到日志，便于复盘开销；
  - backend:
      'openai' —— OpenAI 兼容端点（DeepSeek/Kimi/智谱GLM/本地vLLM），需装 [llm] extras + .env
      'mock'   —— 零依赖离线模拟，让学习框架无需 Key 即可运行；温度越高输出越"发散"
"""
from __future__ import annotations
import logging
import random
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("ygotool.llm")


@dataclass
class Message:
    role: str  # system | user | assistant | tool
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    # assistant 消息携带模型发起的工具调用（ReAct 回灌时用）
    tool_calls: list | None = None


@dataclass
class ChatResult:
    text: str
    usage: dict  # {prompt, completion, total}
    raw: Any = None
    temperature: float = 0.0
    model: str = ""
    reasoning: str = ""  # 推理模型的思维链（GLM-5/Claude/o1 类有）
    tool_calls: list = None  # 模型发起的工具调用；None=无/后端不支持


def _to_tc_dict(tc) -> dict:
    """把工具调用（openai 对象 或 已序列化的 dict）归一成可回灌的 dict。"""
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            name, args = fn.get("name", ""), fn.get("arguments", "")
        else:
            name, args = getattr(fn, "name", ""), getattr(fn, "arguments", "")
        return {"id": tc.get("id"), "type": "function",
                "function": {"name": name, "arguments": args}}
    return {"id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}


def _serialize_tool_calls(tcs: list) -> list[dict]:
    """把 tool_calls（对象或 dict 混合）序列化成可回灌的统一 dict（幂等）。"""
    return [_to_tc_dict(tc) for tc in (tcs or [])]


def _count_tokens(text: str) -> int:
    """token 估算：优先 tiktoken（OpenAI cl100k），不可得时用字符近似。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # 中文按字符、英文粗略 4 字符/token
        return max(1, len(text) // 3)


class _MockBackend:
    """离线模拟。基于规则生成与提示相关的应答，token 用量按真实长度估算。
    温度越高，越可能在末尾追加"发散"内容，便于温度扫描实验看到差异。"""

    POOLS = {
        "ruling": ["根据规则，该效果可以发动。", "此情况属于裁定，不能发动。", "时点已过，无法连锁。"],
        "search": ["相关卡牌：青眼白龙（龙族/光/通常怪兽，3000/2500）。", "未找到完全匹配的卡牌。"],
        "deck": ["建议检索轴：青眼系列，搭配「调和的宝牌」与「太古的白灵龙」。"],
        "generic": ["（模拟回复）已收到问题。"],
    }

    def chat(self, messages: list[Message], *, temperature: float, max_tokens: int,
             model: str, tools=None, response_format=None) -> ChatResult:
        # 取最后一条 user 消息决定主题
        user_text = ""
        for m in reversed(messages):
            if m.role == "user":
                user_text = m.content
                break
        pool = self._pick_pool(messages)
        base = random.choice(self.POOLS[pool])
        # 温度越高，追加越多噪声
        extra = ""
        if temperature > 0.4:
            n = int((temperature - 0.4) * 5)
            extra = " " + "、".join(random.choice(["补充说明", "例外情况", "相关裁定"])
                                   for _ in range(n))
        text = base + extra
        # 尊重 max_tokens：按 token 估算截断
        toks = _count_tokens(text)
        if toks > max_tokens:
            # 粗暴截断到 max_tokens 个"token"对应的字符
            text = text[: max(1, max_tokens * 3)]
        usage = {
            "prompt": sum(_count_tokens(m.content) for m in messages),
            "completion": _count_tokens(text),
            "total": 0,
        }
        usage["total"] = usage["prompt"] + usage["completion"]
        return ChatResult(text=text, usage=usage, raw=None,
                          temperature=temperature, model=model)

    @staticmethod
    def _pick_pool(messages: list[Message]) -> str:
        sys_text = " ".join(m.content for m in messages if m.role == "system").lower()
        if "裁定" in sys_text or "ruling" in sys_text:
            return "ruling"
        if "构筑" in sys_text or "卡组" in sys_text or "deck" in sys_text:
            return "deck"
        if "查卡" in sys_text or "查询" in sys_text or "search" in sys_text:
            return "search"
        return "generic"


class LLMClient:
    def __init__(self, backend: str = "mock", *, base_url: str = "",
                 api_key: str = "", model: str = "mock-1"):
        self.backend = backend
        self.model = model
        if backend == "openai":
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("装 [llm] extras: pip install -e .[llm]") from e
            if not api_key:
                raise ValueError("backend=openai 需要配置 LLM_API_KEY")
            self._client = OpenAI(base_url=base_url, api_key=api_key)
            self._mock = None
        elif backend == "mock":
            self._client = None
            self._mock = _MockBackend()
        else:
            raise ValueError(f"未知 backend: {backend}")

    def chat(self, messages: list[Message], *, temperature: float = 0.2,
             max_tokens: int = 1024, model: str | None = None,
             tools: list | None = None, response_format: dict | None = None,
             thinking: bool | None = None) -> ChatResult:
        """thinking: 推理模型思维链开关（智谱 GLM-5 等）。
        True=开（深思，ruling 用），False=关（检索导向任务用，省 token 防撞上限），
        None=不传，用模型默认。"""
        model = model or self.model
        if self.backend == "mock":
            res = self._mock.chat(messages, temperature=temperature,
                                  max_tokens=max_tokens, model=model,
                                  tools=tools, response_format=response_format)
        else:
            # 序列化消息：assistant 带 tool_calls、tool 角色带 tool_call_id/name
            msgs = []
            for m in messages:
                d: dict[str, Any] = {"role": m.role, "content": m.content or ""}
                if m.tool_calls:
                    d["tool_calls"] = _serialize_tool_calls(m.tool_calls)
                if m.tool_call_id:
                    d["tool_call_id"] = m.tool_call_id
                if m.name:
                    d["name"] = m.name
                msgs.append(d)
            kwargs: dict[str, Any] = {"model": model, "messages": msgs,
                                      "temperature": temperature, "max_tokens": max_tokens}
            if tools:
                kwargs["tools"] = tools
            if response_format:
                kwargs["response_format"] = response_format
            if thinking is not None:
                # 智谱 GLM OpenAI 兼容：thinking 参数走 extra_body
                kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
            resp = self._client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            tcs = getattr(msg, "tool_calls", None)
            res = ChatResult(
                text=msg.content or "",
                usage={
                    "prompt": resp.usage.prompt_tokens,
                    "completion": resp.usage.completion_tokens,
                    "total": resp.usage.total_tokens,
                },
                raw=resp, temperature=temperature, model=model,
                reasoning=getattr(msg, "reasoning_content", None) or "",
                tool_calls=_serialize_tool_calls(tcs) if tcs else None,
            )
        log.info("LLM temp=%.2f model=%s usage=%s", res.temperature, res.model, res.usage)
        return res

    def stream(self, messages: list[Message], *, temperature: float = 0.2,
               max_tokens: int = 1024, model: str | None = None,
               thinking: bool | None = None, tools: list | None = None):
        """流式生成，yield 事件 dict：
        {"type":"reasoning","text":...} / {"type":"content","text":...} /
        {"type":"tool_calls","tool_calls":[...]} / {"type":"usage",...}
        reasoning 仅推理模型且有 thinking_content 时出现；tools 给定时模型可发起工具调用。"""
        from typing import Iterator
        model = model or self.model
        if self.backend == "mock":
            res = self._mock.chat(messages, temperature=temperature,
                                  max_tokens=max_tokens, model=model)
            yield {"type": "usage", "usage": res.usage}
            # 按词切分模拟流式
            text = res.text
            step = max(1, len(text) // 12)
            for i in range(0, len(text), step):
                yield {"type": "content", "text": text[i:i + step]}
            return
        # 序列化消息：assistant 带 tool_calls、tool 角色带 tool_call_id/name
        msgs = []
        for m in messages:
            d: dict[str, Any] = {"role": m.role, "content": m.content or ""}
            if m.tool_calls:
                d["tool_calls"] = _serialize_tool_calls(m.tool_calls)
            if m.tool_call_id:
                d["tool_call_id"] = m.tool_call_id
            if m.name:
                d["name"] = m.name
            msgs.append(d)
        kwargs: dict[str, Any] = {
            "model": model, "messages": msgs,
            "temperature": temperature, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
        if thinking is not None:
            kwargs["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
        stream = self._client.chat.completions.create(**kwargs)
        usage = None
        # 累积流式 tool_calls 分片（按 index 拼 id/name/arguments）
        tc_acc: dict[int, dict] = {}
        try:
            for chunk in stream:
                if chunk.usage:
                    usage = {
                        "prompt": chunk.usage.prompt_tokens,
                        "completion": chunk.usage.completion_tokens,
                        "total": chunk.usage.total_tokens,
                    }
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if not delta:
                    continue
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    yield {"type": "reasoning", "text": rc}
                if delta.content:
                    yield {"type": "content", "text": delta.content}
                # 累积工具调用分片
                for ptc in (getattr(delta, "tool_calls", None) or []):
                    idx = ptc.index if ptc.index is not None else 0
                    slot = tc_acc.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                    if getattr(ptc, "id", None):
                        slot["id"] = ptc.id
                    fn = getattr(ptc, "function", None)
                    if fn:
                        if fn.name:
                            slot["name"] += fn.name
                        if fn.arguments:
                            slot["arguments"] += fn.arguments
        except Exception as e:
            # 智谱流式中途断连等：优雅收尾，让上层发 error 事件而非裸异常
            yield {"type": "error", "message": f"LLM 流式中断: {e}"}
            return
        if tc_acc:
            tcs = [{"id": v["id"], "type": "function",
                    "function": {"name": v["name"], "arguments": v["arguments"]}}
                   for v in tc_acc.values()]
            yield {"type": "tool_calls", "tool_calls": tcs}
        if usage:
            yield {"type": "usage", "usage": usage}

