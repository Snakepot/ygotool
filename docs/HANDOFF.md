# ygotool 工作交接（2026-09-14）

> 供新对话续接用。本文 + `.claude/skills/ygotool/SKILL.md` + `docs/agent-concepts-review.md` + `README.md` 是完整知识库。新会话直接 `claude --continue` 或 `claude`（记忆自动加载）。

## 一、项目定位
- **路径**：`E:\emergency\正经工作\ygotool`（Windows，PowerShell + Git Bash）
- **目标**：基于 https://ygocdb.com API 的本地游戏王卡牌工具 + Agent；**学习型项目**（理解 RAG切分/Few-shot/Temperature/Token控制，AB 层推理参数+上下文工程，**不做 C 层权重微调**）
- **LLM 后端**：智谱 GLM-5.2（用户有 2000w token 包），`.env` 已配。**GLM-5.2 是推理模型**（有 reasoning_content，算进 completion token，max_tokens 要留推理预算）
- **技术栈**：Python（标准库优先，extras：[llm]/[web]/[vec]/[embed]）；FastAPI + 原生 HTML/JS（无构建）；sqlite（无 sqlite-vec，暴力检索够用——卡池规模有限）

## 二、项目结构
```
ygotool/
├── config.py / .env / pyproject.toml / cli.py / ygotool.bat / README.md
├── ygocdb/        数据层：client(HTTP) / codec(位掩码解码) / store(sqlite缓存+实体链接) / effect_parser(PSCT解析)
├── rag/           RAG：chunker(3切分器) / embedder(TF-IDF|openai|local) / index(sqlite向量) / retriever
├── agent/         llm(openai兼容+mock, stream, thinking) / prompts(persona+原则+few-shot) / budget(优先级裁剪) / tools(ToolRuntime) / agent(CardAgent)
├── web/server.py  FastAPI（REST + SSE 流式 chat，心跳保活）
├── web/static/    index.html / app.js / style.css（原生，无构建）
├── examples/      01切分对比 / 02few-shot / 03温度 / 04预算
├── cache/         cards.zip / cards.db / idx_main.db / details 缓存
├── docs/agent-concepts-review.md   概念地图+面试题+数据结构化边界(§13)
└── .claude/skills/ygotool/SKILL.md 工作端 agent skill（含踩坑手册17条）
```
**数据流**：cards.zip(MD5增量) → sqlite cards 表 → RAG 索引(idx_main.db) → CardAgent(_prepare: 工具+预算 → LLM) → trace。

## 三、核心思路与工具
- **数据层**：ygocdb API 全量卡池 zip + MD5 增量同步到 sqlite；`codec.py` 解码 YGOPro 位掩码（**race 非标准**，从全量卡池反推；attribute/type 标准；**Link 卡 level 是 Link 数值非箭头位掩码**，箭头只在 text.types）。`effect_parser.py` 按 ①② 分条 + "才能发动"分离 cost/效果体。
- **RAG**：三种切分器（整卡/按①②效果/FAQ Q-A）；默认 TF-IDF（零依赖，**跨语言不行**——FAQ 是日文）；纯 Python 暴力余弦，不需要 sqlite-vec。
- **Agent**：`CardAgent` **确定性管线**（非 ReAct，因 mock 后端不支持原生 tool-calling）。工具增强降幻觉——必经工具取数，不靠 LLM 记忆。trace 全程可见（工具链/预算/token）。预算优先级裁剪：system > 问题 > 历史(最新) > few-shot > 检索chunk。
- **人设 A/B**：strict（裁定官，结论先行+引用式）/ teach（教学向导，多解释原理）。
- **原则库**（ruling 系统提示内）：cost≠效果、取对象≠发动后挑选、触发字面性（丢弃≠送去墓地≠cost）、90X 后 PSCT 统一但"未统一≠裁定不同"（判措辞非日期）、原则+FAQ 双轨。
- **搜索三维度**：`search_card`(名 LIKE) / `search_by_effect`(PSCT 解析+同义词+主动/被动区分+全量两档:现代/未确认) / `search_by_series`(LCS 提 token+卡名 LIKE 全量)。意图路由：list_intent(效果)/series_intent(系列)。
- **Web**：FastAPI SSE 流式（异步+5s 心跳保活，防推理模型思考久掐断）+ 原生前端（Markdown 渲染/思考折叠/模态详情/卡牌#id 可点/多轮历史/停止/深色/右栏折叠）。

## 四、完成情况
✅ **阶段 1-6 全部落地**：数据层+codec、RAG、LLM/Few-shot、Agent 编排+预算、CLI+README、Web 前端（FastAPI+原生JS+流式）。
✅ **核心功能验证**：
- 卡牌查询（单卡/按效果/按系列，全量返回）
- 裁定问答（实体链接+FAQ+原则演绎，带 trace/引用）
- cost/效果区分（魔力净化=cost、刻魔的咏圣=效果，都答对）
- 按效果找卡（"舍弃手卡"全量两档，主动排除被动触发型）
- 多轮对话、流式、人设 A/B
✅ **踩坑修复**：sqlite 跨线程、SSE 心跳+ensure_future、reasoning 吃预算、死循环、检索反噬、race 编码、Link level、citation id、枚举 cap 截断、.bat 的 nul、modal hidden 等（详见 SKILL.md §踩坑速查）。
✅ **文档**：README、docs/agent-concepts-review.md（概念地图+16 面试题+数据边界§13）、`.claude/skills/ygotool/SKILL.md`（工作 skill+17 坑）。
✅ **运维**：`ygotool.bat`（toggle 开关，按端口 8000）。

## 五、已知限制 / 待办（按优先级）
1. **RAG 召回弱**（高）：索引只覆盖前 300 张卡、TF-IDF 跨语言→裁定靠"恰好问到那张卡自己有 FAQ"，跨卡裁定漏。**修法**：全量索引 + 多语言 embedding（智谱 embedding 或 bge-multilingual）。暴力检索够用，不需 sqlite-vec。
2. **裁定动作本体缺口**（高）："因效果送入墓地/因对手离开场上"等**原因细分**无结构化字段，靠 FAQ+原则。**修法**：建动作本体（送入墓地来源/原因 taxonomy），PSCT 解析时打标。
3. **实体链接错字**（中）："手牌/手札"等错字仍漏（子串匹配不同字符）。**修法**：模糊匹配或让 LLM 先纠错。
4. **deck 任务弱**（中）：TF-IDF 召不回主题卡。**修法**：同 #1，全量索引+真 embedding。
5. **原则库单薄**（中）：只 3-4 条在 prompt 里，未成体系。**修法**：curate 20-50 条权威原则（错一条污染大，需权威来源）。
6. **无评测指标**（中）：examples 只展示趋势，无"正确率/命中率"量化。**修法**：固定测试集+accuracy harness。
7. **FAQ 未批量索引**（低）：现仅按卡 fetch_detail；跨卡 FAQ 索引未建（#1 含）。
8. **原生 function-calling ReAct 未接**（低）：当前确定性管线；接真模型可换 ReAct（trace 结构不变）。

## 六、记忆文件（自动加载）
`C:\Users\Eric\.claude\projects\E--emergency------ygotool\memory\` 下：
- `ygotool-project-overview.md` — 目标/范围/LLM后端/裁定路线
- `ygocdb-race-encoding-nonstandard.md` — race 非标准编码
- `ygo-ruling-psct-distinction.md` — 文本统一≠裁定对错；90X卡包后统一；判措辞非日期
- 用户是**游戏王裁定行家**，涉及裁定判断采信其纠正。

## 七、新对话怎么续
- `claude --continue`（接着本次）或 `claude`（新对话，记忆自动加载）
- 服务：`.\ygotool.bat`（toggle）
- 知识库：先读 `.claude/skills/ygotool/SKILL.md`（含踩坑手册）+ 本文件 + `docs/agent-concepts-review.md`
- **下一步候选**（按 §五 优先级）：先做"全量索引 + 多语言 embedding"（#1，修 ruling/deck 召回）或"裁定动作本体"（#2）；或先写 `ygotool-rulings` 领域 skill；或建评测框架（#6）。

## 八、关键设计决策（勿推翻）
- **后端可热切**：LLM/embedder 的 base_url+model 可配，对上层透明。
- **TF-IDF 起步**：零依赖离线可跑；真上量切多语言 embedding，接口不变。
- **工具增强降幻觉**：Agent 答卡牌信息必经工具取数。
- **暴力检索够用**：卡池 ~1.5 万、年增几百，不需 sqlite-vec/FAISS。
- **原则+FAQ 双轨**：原则管可推广的，FAQ 管不可还原的；两者都喂推理模型。
- **trace 全程可见**：调参/复盘的基础。
- **显示优先 text.types**：`data` 位掩码解码仅教学，UI 用权威字符串。
