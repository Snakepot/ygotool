"""学习示例 02：Few-shot vs Zero-shot（★ Few-shot 学习点）。

对同一组测试题，分别注入 0 / 1 / 2 个 few-shot 示例，对比：
  - 输出文本（mock 下基本不变；真模型下应随 few-shot 变好）
  - token 用量（few-shot 越多，prompt token 越多——这是 few-shot 的成本）

跑法：python examples/02_fewshot_vs_zeroshot.py
后端：默认 mock（离线可跑，看 token 成本曲线）；配好 .env 后改 backend='openai' 看质量差异。
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from agent.llm import LLMClient
from agent.prompts import build_messages

TASK = "ruling"
TESTS = [
    "异色眼灵摆龙被破坏时能发动效果吗？",
    "青眼白龙能被「死者苏生」特殊召唤吗？",
]
SHOTS_SWEEP = [0, 1, 2]


def make_client(settings) -> LLMClient:
    if settings.llm_api_key:
        return LLMClient("openai", base_url=settings.llm_base_url,
                          api_key=settings.llm_api_key, model=settings.llm_model)
    print("[note] 未配置 LLM_API_KEY，使用 mock 后端（仅演示 token 成本曲线，质量差异需接真模型）")
    return LLMClient("mock")


def main() -> None:
    s = get_settings()
    client = make_client(s)
    rows = []
    print(f"任务: {TASK} | 后端: {client.backend}\n")
    print(f"{'问题':<28}{'shots':<7}{'prompt':<8}{'compl':<8}{'total':<8}输出")
    print("-" * 90)
    for q in TESTS:
        for n in SHOTS_SWEEP:
            msgs = build_messages(TASK, q, num_few_shots=n)
            res = client.chat(msgs, temperature=0.0, max_tokens=s.default_max_tokens)
            u = res.usage
            out = res.text[:30].replace("\n", " ")
            print(f"{q[:26]:<28}{n:<7}{u['prompt']:<8}{u['completion']:<8}{u['total']:<8}{out}")
            rows.append({"question": q, "shots": n, **u, "output": res.text})

    out_csv = Path(__file__).parent / "02_fewshot_vs_zeroshot.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "shots", "prompt", "completion", "total", "output"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写 {out_csv}")
    print("\n学习观察：")
    print(" - few-shot 越多 → prompt token 越高（成本曲线），这是 few-shot 的代价；")
    print(" - 接真模型后：0-shot 易跑题/格式乱，加 shot 后格式与准确率应提升——质量收益。")


if __name__ == "__main__":
    main()
