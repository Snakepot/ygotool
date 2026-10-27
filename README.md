# ygotool

基于 [ygocdb.com](https://ygocdb.com) 官方 API 的本地卡牌工具 + 游戏王卡牌 Agent。
同时是一个**学习型项目**：把 RAG 切分 / Few-shot / Temperature / Token 控制 / 微调数据导出
做成可演示、可调参、能看清内部行为的模块。

## 功能

- **本地工具**：全量卡池 MD5 增量同步、离线按名/密码/cid 查询、YGOPro 位运算字段解码、禁限卡表。
- **RAG 层**：三种切分器（整卡/按效果①②/FAQ Q-A）、可切换 embedder（零依赖 TF-IDF / OpenAI 兼容 / 本地）、向量检索。
- **Agent**：三任务（自然语言查卡 / 裁定问答 / 卡组构筑）、工具增强编排（降幻觉）、Token 预算管理。

## 安装

```bash
pip install -e .            # 基础（仅标准库即可运行阶段1/2）
pip install -e ".[llm]"     # 加 OpenAI 兼容 LLM 客户端
pip install -e ".[vec]"     # 加 sqlite-vec（规模化检索）
pip install -e ".[embed]"   # 加本地 sentence-transformers
```

配置（可选，接真模型时需要）：

```bash
cp .env.example .env
# 填 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL（DeepSeek/Kimi/智谱GLM/本地vLLM 均可）
```

## 命令

```bash
ygotool refresh                         # 同步本地卡池（MD5 增量）
ygotool search 青眼                     # 按名称查卡（离线）
ygotool card 89631139 [--faq]          # 按密码取详情 + 解码
ygotool limits                          # 禁限卡表概览
ygotool build-index [--chunker effect] [--n 500]   # 建 RAG 索引
ygotool agent "问题" [--task ruling]    # 问答（trace 可见工具链与 token）
```

## 学习点导览

| 学习点 | 看哪里 | 怎么观察 |
|---|---|---|
| **RAG 切分** | `rag/chunker.py` + `examples/01` | 三切分器 top-k 得分与命中边界对比 |
| **Few-shot** | `agent/prompts.py` + `examples/02` | `num_few_shots` 0/1/2 的 token 成本曲线 |
| **Temperature** | `agent/llm.py` + `examples/03` | 温度扫描的多样性与 token 变化 |
| **Token 控制** | `agent/budget.py` + `examples/04` | 预算扫描看检索被裁到几个 |
| **工具降幻觉** | `agent/tools.py` + `agent/agent.py` | Agent 必先调工具拉真实数据再答 |
| **微调数据导出**（预留） | `examples/` | AB 层是 C 层微调的"前一步"：数据构造→few-shot→评测链路在此可复现 |

### 运行示例（默认零依赖、离线可跑）

```bash
python examples/01_chunking_compare.py     # 切分对比
python examples/02_fewshot_vs_zeroshot.py   # few-shot 成本曲线
python examples/03_temperature_sweep.py     # 温度扫描
python examples/04_token_budget.py          # 预算扫描
```

> 未配置 `LLM_API_KEY` 时自动用 `mock` 后端：能看 token 成本/趋势曲线，
> 但输出是模拟的。配好 `.env` 后同一份脚本自动接真模型看真实质量。

## 架构

```
config.py            配置（.env + 默认值）
ygocdb/              数据层：client(HTTP) / codec(位运算解码) / store(sqlite缓存)
rag/                 RAG：chunker / embedder / index / retriever
agent/               LLM 客户端 / 提示词+few-shot / 工具 / 预算 / 编排
cli.py               命令行入口
examples/            学习示例（每个隔离一个技术点）
cache/               cards.zip / cards.db / idx_main.db（运行时产物）
```

## 关键设计决策

- **后端可热切**：LLM 与 embedder 都做成 `base_url/model` 可配，换厂商/换模型改一行；
  对上层透明，`Retriever`/`Agent` 代码不动。
- **TF-IDF 起步**：零依赖、离线可跑，让切分/检索的学习效果立刻可见；
  真上量时切 OpenAI 兼容 embedding 或 sentence-transformers，接口不变。
- **工具增强降幻觉**：Agent 回答卡牌信息必经工具取数，不靠 LLM 记忆。
- **trace 全程可见**：每次问答输出调了哪些工具、保留几个 chunk、token 用量、预算报告。

## 说明

- ygocdb `data.race` 编码**非标准 YGOPro**（龙族=0x2000 而非 0x80000），
  `codec.py` 的种族表是从全量卡池（14295 张）反推的权威映射。
- `text.types` 已是人类可读字符串（如 `[怪兽|通常] 龙/光`），`codec` 模块作教学补充。
- 卡牌 FAQ 为官方数据库日文原文、混 HTML；`FaqChunker` 已做剥离。
- 遵守站长善意：全量卡池走 MD5 增量、单卡详情带磁盘缓存避免高频请求。
