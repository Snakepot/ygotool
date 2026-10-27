---
name: ygotool
description: 在 ygotool 项目（E:\emergency\正经工作\ygotool，基于 ygocdb API 的游戏王卡牌 Agent + 本地工具）上做开发、运维、调试、扩展时使用。涵盖架构导航、启停操作、踩坑速查、扩展模式、编码约定。当用户在这个项目里改代码、跑服务、查 bug、加功能、调参时触发。
---

# ygotool 工作手册（给工作端 agent）

本手册用于在本项目上高效干活。先看架构，再查操作，遇坑查 §踩坑速查，加功能查 §扩展模式。

## 架构地图
```
ygotool/
├── config.py / .env          配置（LLM_BASE_URL/KEY/MODEL、温度、max_tokens、max_context）
├── cli.py                    命令入口：serve/refresh/search/card/limits/build-index/agent
├── ygotool.bat               服务开关脚本（toggle，按端口 8000）
├── ygocdb/                   数据层
│   ├── client.py            ygocdb HTTP（search/card/cardset/limits/releaseDates/cards.zip）
│   ├── codec.py             YGOPro 位掩码解码（race 表非标准，从全量卡池反推）
│   ├── store.py             sqlite 缓存（cards 表 + details 表，find_cards_in_text 实体链接）
│   └── effect_parser.py     PSCT 解析（按 ①② 分条、"才能发动"分离 cost/效果体）
├── rag/                      RAG：chunker(3 切分器)/embedder(TF-IDF|openai|local)/index(sqlite向量)/retriever
├── agent/                    llm(openai兼容+mock, stream, thinking开关)/prompts(persona+原则+few-shot)/budget(优先级裁剪)/tools(ToolRuntime)/agent(CardAgent)
├── web/server.py            FastAPI（REST + SSE 流式 chat，心跳保活）
├── web/static/              原生 HTML/JS/CSS（无构建）
├── examples/01-04           学习示例（切分/few-shot/温度/预算）
└── cache/                    cards.zip/cards.db/idx_main.db/details 缓存
```
数据流：cards.zip(MD5 增量)→sqlite cards 表→RAG 索引→CardAgent(_prepare: 工具+预算→LLM)→trace。

## 操作节奏
- 启停服务：`.\ygotool.bat`（toggle）/ `ygotool.bat start|stop|status`；或 `python cli.py serve`。Web 在 http://127.0.0.1:8000
- 改了 Python 代码 → **必须重启服务**才生效（无热重载）；改了 static/* → 浏览器 **Ctrl+F5** 清缓存
- 重启前先杀旧进程：`netstat -ano | grep :8000` 取 PID → `taskkill //F //PID <pid>`
- 刷新卡池：`python cli.py refresh`（MD5 一致则跳过）
- 建索引：`python cli.py build-index [--chunker effect] [--n 500]` → cache/idx_main.db（ruling/deck 依赖它）
- 跑示例：`python examples/0X_*.py`（01 切分对比/02 few-shot/03 温度/04 预算）
- 中文输出乱码：`export PYTHONIOENCODING=utf-8`；脚本里 `sys.stdout.reconfigure(encoding='utf-8')`
- `.env` 配 GLM：base_url=`https://open.bigmodel.cn/api/paas/v4/`，model=`glm-5.2`

## 踩坑速查（症状 → 根因 → 修法）★核心★
1. **`SQLite objects created in a thread...`**：FastAPI 请求在子线程，sqlite 连接主线程建。→ 连接开 `check_same_thread=False`（store.py、rag/index.py），写操作加 `threading.Lock`（store.fetch_detail 的 INSERT）。
2. **SSE `TypeError: network error` 中断**：推理模型思考数十秒无数据，连接被掐。→ `api_chat` 改异步 + 每 5s 发 `: ping` 心跳；`llm.stream` try/except 智谱断连发 error 事件。**注意**：`loop.run_in_executor` 返回 **Future 不是 coroutine**，别用 `asyncio.create_task`（会 TypeError 秒断），用 `asyncio.ensure_future`。
3. **回答内容空/截断**：推理模型 `reasoning_content` 吃满 max_tokens，content 没机会生成。→ max_tokens 调到 8192；非 ruling 任务（search/deck/列举）关 thinking（`extra_body={"thinking":{"type":"disabled"}}`）。
4. **推理死循环（翻满 max_tokens 全垃圾）**：检索不到时推理模型 flail 反复打转。→ 补检索（search_by_text/effect/series）；列举型查询关 thinking。
5. **"检索反而帮倒忙"**：strict grounding + 不完整检索 → 模型放弃自身正确参数知识迁就有缺陷检索（答错）。→ 补全检索（跨卡 FAQ 索引+多语言 embedding）；或放宽 grounding 为"检索优先但不全时允许演绎标置信度"。
6. **TF-IDF 跨语言不命中**：FAQ 日文、查询中文，字面不重叠。→ 换多语言 embedding（bge-multilingual 或智谱 embedding）；TF-IDF 仅作学习起点。
7. **race 解码错（龙族等）**：ygocdb `data.race` **非标准 YGOPro**（龙=0x2000 不是 0x80000）。→ 用 `codec.RACE` 表（从全量卡池反推），别照抄 constant.lua。attribute/type 是标准的。
8. **Link 卡 level 解码错**：`data.level` 存的是 **Link 数值**（LINK-2→2），**不是箭头位掩码**；箭头只在 `text.types` 字符串。→ `decode_level` Link 分支返回 `{link: lv, markers: []}`；显示优先用 `text.types`。
9. **citation 点不开/对不上**：引用混用 cid/id。→ `budget._render` 统一用密码 `id`（`meta["id"]` 优先），前端 `卡牌#<id>` 可点开详情。
10. **枚举查询漏卡**：`find_cards_in_text`/search 有 cap（5 或 top-k）+ 同分 tie-break 随机 → 漏。→ 枚举类用 `search_by_effect`/`search_by_series`（全量返回不截断），不给 LLM 整块大名单（超预算被整体丢）→ 改"总数+前30样例"，全量走浏览端点。
11. **`find_cards_in_text` 漏简称/子串**：片段匹配只对带分隔符卡名（神影依·米德拉什→米德拉什）有效，无分隔名（青眼白龙 vs 青眼）和错字（手牌/手札）漏。→ 已用 `SequenceMatcher` 最长公共子串改进；错字仍需模糊匹配或让 LLM 纠错。
12. **`.bat` 报错 `>/dev/null` 不是命令**：linter 把 cmd 的 `>nul` 改成 bash 的 `>/dev/null`。→ .bat 里用 `>nul 2>&1`；测 .bat 用 PowerShell `.\ygotool.bat`（别用 bash `./x.bat`，会吃掉 `^|` 转义）。
13. **后台服务没保住**：bash `cmd &` 会话结束进程没保。→ 用 `run_in_background=true` 或 `ygotool.bat`（`start /MIN`）。
14. **`from openai import OpenAI` 报缺**：未装 `[llm]` extras。→ `pip install -e ".[llm]"`（web 同理 `.[web]`）。
15. **modal 关不掉铺满屏**：`.modal{display:flex}` 优先级高于 `[hidden]` 的 `display:none`。→ 加 `.modal[hidden]{display:none !important}`。
16. **Windows 中文路径/GBK**：bash→cmd 传中文路径会乱。→ 测 .bat 用 PowerShell；脚本输出加 `PYTHONIOENCODING=utf-8`。
17. **search 任务超时 `WinError 10060`**：`_link_and_fetch` 在线拉 FAQ 被限流。→ search 用 `with_faq=False`（仅本地），ruling 才联网拉 FAQ（带 try/except 回退）。
18. **Web 检索分数失真（embedder/索引维度错配）**：`web/server.py` 的 `_load_retriever` 曾各自实现成固定 `Embedder("tfidf")` 加载索引，而索引是按 embedding-3(1024 维) 建的 → 查询向量与库内维度不同、点积截断成垃圾分数（裁定/构筑问答弱）。→ 统一用 `rag.retriever.load_retriever(settings)`（读 index_meta 构造匹配 embedder，tfidf 才 fit）——cli 与 web 共用，别各写一份。
19. **ReAct 的 thinking 没真正开**：原逻辑 `thinking=th if force_final else False`，而 force_final 只在撞工具调用上限时置真 → 模型自然收敛的最终答也在 thinking 关闭下产出。→ 改为「首轮路由关、拿到工具结果的后续轮开」：`th if (force_final or it > 0) else False`（`_react_ask`/`_react_stream` 两处）。
20. **search_by_action 参数名**：方法参数是 `reason`，`agent._prepare` 枚举分支曾误传 `reason_filter=` → 走枚举意图时 TypeError 崩（mock/确定性路径；schema 与 server 传的是对的）。→ 统一传 `reason=`。
21. **deck 分支 `hits` 未绑定**：`hits` 只在 `if not list_answered:` 内赋值，候选卡循环却写在外 → 枚举意图命中时 `UnboundLocalError`。→ 循环收进同一守卫内。
22. **本体缺「解放」（tribute）等高频动作**：用户粘贴原文「把自己的手卡·场上1只怪兽解放才能发动」→ `search_by_action` 映射为空返回 0；子串兜底又给 3509 张噪音。→ ontology 补 `release` 动作 + `map_phrase_to_actions` 认 解放/献祭/祭品；`tag_actions` 跳过否定句并新增完整条款 `clause` 字段；`search_by_action` 加 `source` 过滤（组合来源如「手卡·场上」要求区域词都出现）。
23. **粘贴原文找不到卡**：子串/词条匹配搞不定整句措辞。→ 新增 `search_effect_phrase(phrase)`：原文与效果文本同样规范化（去标点）后按最长连续匹配（≥4 字）+ 4 字片段覆盖率排序，cost 段参与匹配；REACT_RULE 教模型「原文→动作+来源，cost 在才能发动前」。
24. **特异度没落地（P03）**：「从卡组将等级3怪兽送入墓地」实际命中 339 张（含唯一限定 3星的基路伯），但基路伯排在 126 位、`_render_list` 只给前 15 → LLM 误答"未找到明确限定的卡"。→ `_query_modifiers`（等级/属性/种族）+ 每条结果 `specific` 标记排序 + `_render_list` 分组「★明确限定 / 通用」，精确匹配永远先呈现。
25. **来源绑定粒度太粗**：「解放手卡」305 张里大量误召（次元魔法：那只怪兽解放，从手卡把…特殊召唤；死皇帝之陵墓：**不用**解放…手卡1只怪兽；念动力脉冲/熔岩龙：手卡出现在其它分句）。→ 新增 `tag_action_units`：cost/效果体各自按连词切动作单元，zone 只在单元内提取；否定词扩到 不用/无需/不需要/不进行；代词句只继承"区域+数量+「目标」"句式（`extract_object_source`）的 zone。search_by_action 改用它。
26. **查询归一化**：等级3/N星/N星级 → N星；「从卡组」不匹配「额外卡组/融合卡组」（负向断言 `(?<!额外)(?<!融合)卡组`）；source 留空时自动从短语抽 zone 词。
27. **别名表接线**：`agent/data/aliases.md` 由 `agent/data_loader.load_aliases/apply_aliases` 解析（格式：`- 别名A|别名B: 规范`、`regex: 模式 -> 替换`、`⏸` 开头跳过、`- （` 注释跳过）；`tools._normalize_query` 先跑别名再跑等级正则；`search_by_action/search_by_effect/search_effect_phrase` 入口都归一化。坑：别名 `通常怪→通常怪兽` 会撞已写好的「通常怪兽」变成「怪兽兽」→ 用 `(?!兽)` 负向断言；精确匹配组在 `_render_list` 要列全（上限 30，否则排第 23 位的目标卡 LLM 看不到）。

## 扩展模式
- **加工具**：`ToolRuntime` 加方法（返回 `(text, trace)`）→ `agent._prepare` 按意图路由调用 → `trace.tools.append(...)` →（可选）`TOOLS` JSON schema 供原生 function-calling →（可选）`web/server.py` 加 `/api/xxx` 端点 + `app.js` 调用。
- **加切分器**：`rag/chunker.py` 加类（输出 `Chunk(text, meta)`）→ `CHUNKERS` 注册 → `build_index`/`build_faq_index` 用。
- **加原则**：`agent/prompts.py` 的 `PRINCIPLES_BLOCK` 加条目（ruling 系统提示自动含）。
- **加人设**：`PERSONAS` 字典加 → `build_system(task, persona)` 自动支持。
- **加推理参数**：全链路透传——`agent.ask/ask_stream/_prepare` → `budget.fit` → `build_messages`（参考 persona/history/max_context/thinking 的透传）。
- **新搜索维度**：参考 `search_by_effect`（PSCT 解析+同义词+主动/被动+全量两档）或 `search_by_series`（LCS 提取 token+name LIKE 全量）。

## 编码约定
- **零依赖优先**：标准库能做就不引包；额外依赖放 `[extras]`（llm/web/vec/embed），mock 后端兜底让无 Key 也能跑。
- **对上层透明**：`Embedder`/`LLMClient` 暴露统一方法，换后端只改构造一行。
- **trace 全程可见**：工具链/预算/token 用量都进 trace，可复盘。
- **尊重数据源**：cards.zip 走 MD5 增量、FAQ 磁盘缓存、`YgocdbClient(min_interval=0.3)` 限流。
- **渐进披露**：本 skill 精简，深度内容查 `README.md` 与 `docs/agent-concepts-review.md`（含概念地图、面试题、§13 数据结构化边界）。
- **裁定领域知识**（排查裁定答案对错时）：cost≠效果、取对象≠发动后挑选、触发字面性（丢弃≠送去墓地≠cost）、90X 后 PSCT 统一但"未统一≠裁定不同"（判措辞非日期）、原则+FAQ 双轨。原则原文在 `agent/prompts.py`。

## 边界
- 只做 AB 层（推理参数 + 上下文工程），不做 C 层权重微调（用闭源 API）。
- 动作/原因类细分（"因效果送入墓地"）无结构化字段，靠 FAQ+原则；身份类（种族/类型/星级）位掩码精确。
- mock 后端不支持原生 tool-calling → 当前用确定性管线，接真模型可换 ReAct（trace 结构不变）。
