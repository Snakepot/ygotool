"""学习示例 03：Temperature 扫描（★ Temperature 学习点）。

对同一道题，在 temperature ∈ {0,0.3,0.7,1.0} 各跑 N 次，统计：
  - 输出多样性（不同答案数 / N）
  - token 用量
温度越高 → 输出越发散、越不稳定；温度=0 理论上确定性。

跑法：python examples/03_temperature_sweep.py
后端：默认 mock（能看到温度↑→输出更发散）；配好 .env 后改 backend='openai' 看真模型发散。
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
QUERY = "异色眼灵摆龙被破坏时能发动效果吗？"
TEMPS = [0.0, 0.3, 0.7, 1.0]
N = 3  # 每个温度跑几次


def make_client(settings) -> LLMClient:
    if settings.llm_api_key:
        return LLMClient("openai", base_url=settings.llm_base_url,
                          api_key=settings.llm_api_key, model=settings.llm_model)
    print("[note] 未配置 LLM_API_KEY，使用 mock 后端（演示温度↑→发散↑）")
    return LLMClient("mock")


def main() -> None:
    s = get_settings()
    client = make_client(s)
    print(f"任务: {TASK} | 后端: {client.backend} | 每温度跑 {N} 次\n")
    print(f"{'temp':<7}{'多样性(x/N)':<14}{'avg_prompt':<11}{'avg_compl':<11}{'avg_total':<11}样例输出")
    print("-" * 90)

    rows = []
    for t in TEMPS:
        outs, totals, prompts, comps = [], [], [], []
        for _ in range(N):
            msgs = build_messages(TASK, QUERY, num_few_shots=1)
            res = client.chat(msgs, temperature=t, max_tokens=s.default_max_tokens)
            outs.append(res.text)
            totals.append(res.usage["total"])
            prompts.append(res.usage["prompt"])
            comps.append(res.usage["completion"])
        distinct = len(set(outs))
        ap = sum(prompts) / N
        ac = sum(comps) / N
        at = sum(totals) / N
        print(f"{t:<7}{distinct}/{N:<13}{ap:<11.0f}{ac:<11.0f}{at:<11.0f}{outs[0][:30].replace(chr(10),' ')}")
        rows.append({"temp": t, "distinct": distinct, "n": N,
                     "avg_prompt": ap, "avg_completion": ac, "avg_total": at})

    out_csv = Path(__file__).parent / "03_temperature_sweep.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["temp", "distinct", "n",
                                          "avg_prompt", "avg_completion", "avg_total"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写 {out_csv}")
    print("\n学习观察：")
    print(" - temp=0 输出应高度一致（确定性）；temp↑ → 多样性↑、稳定性↓；")
    print(" - 实践：事实问答/裁定用低温度(0~0.3)；创意/发散用高温度(0.7~1.0)。")


if __name__ == "__main__":
    main()
