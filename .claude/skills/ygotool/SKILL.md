---
name: ygotool
description: 在 ygotool 项目（基于 ygocdb API 的游戏王卡牌 Agent + 本地工具）上做开发、运维、调试、扩展时使用。涵盖架构导航、启停操作、踩坑速查、扩展模式、编码约定。当用户在这个项目里改代码、跑服务、查 bug、加功能、调参时触发。
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
10. **枚举查询漏卡**：`find_cards_in_text`/search 有 cap（5 或 top-k）+ 同分 tie-break 随机 → 漏。→ 枚举类用 `search_by_effect`/`search_by_series`（全量返回不截断），不给 LLM 整块大名单（超预算被整体丢）→ 改"总数+样例"，全量走浏览端点（P1 后名单 ≤60 张全量渲染，超过才截断）。
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
27. **别名表接线**：`agent/data/aliases.md` 由 `agent/data_loader.load_aliases/apply_aliases` 解析（格式：`- 别名A|别名B: 规范`、`regex: 模式 -> 替换`、`⏸` 开头跳过、`- （` 注释跳过）；`tools._normalize_query` 先跑别名再跑等级正则；`search_by_action/search_by_effect/search_effect_phrase` 入口都归一化。坑：别名 `通常怪→通常怪兽` 会撞已写好的「通常怪兽」变成「怪兽兽」→ 用 `(?!兽)` 负向断言；精确匹配组在 `_render_list` 要列全（P1 后名单 ≤60 张全量渲染；旧上限 30 会把排第 23 位的目标卡截掉）。
28. **"来源"被误当作查询区 / 次精确档（三档分级）**：用户问「从手卡把水属性怪兽送去墓地」时，① 查询短语里的「墓地」是**目的地**不是来源（`_zone_words_in_phrase` 曾把目的地算进来）；② 卡文「手卡·卡组」「手卡·场上（表侧表示）」离动作词太远，`手[牌卡][^。]{0,8}` 窗口匹配不到 → 波塞德拉深渊/轰海皇被漏。→ 用 `ontology.source_zones_from_phrase`（动词 `VERB_WORDS` 之前的片段才当来源）和 `first_verb_pos`（不用含来源的整条正则起点）；`tag_action_units` 记录每单元 `zones`（动作词前片段；动词在句首时仅解放/丢弃类取动词后区域词），代词句回退 clause 的 cost 区域；`send_hand_to_gy` 窗口放宽到 0,20 并新增「手卡·卡组/手卡·场上」组合匹配；连词表加「或/或者」（防「从手卡把钢核送去墓地**或**把手卡1只水属性怪兽给对方观看」这类跨动作串味）。三档：★精确（限定一致+来源一致）/◆次精确（限定一致但来源更宽，如手卡·卡组）/通用；`_render_list` 分三组，提示词要求回答补「此外，也有…来源更宽的卡」。
29. **来源区被修饰语污染（「额外卡组」≠送墓来源）**：哀悼傀儡/炼狱的灾天「把最多有从额外卡组特殊召唤的对方场上的怪兽数量＋1张的…从卡组送去墓地」——「额外卡组」只是「特殊召唤」的来源，`first_verb_pos` 截在修饰语动词上 → 整句被当成「从额外卡组送墓」召回（query 噪音）。→ `effect_parser._SRC_MODIFIER_RE` 先剔「从X特殊召唤**的**」修饰语再识别动作与来源区（只剔带「的」的修饰；`从X特殊召唤1只怪兽` 是真实动作不能剔）。顺带：`_render_list` 名单 ≤60 张时全量渲染（原来 30/15 截断导致 LLM 只能列出前 30 张，与评测金标错位）；大名单仍截断、完整名单走左栏。
30. **「效果无效化 / 不能把效果发动」查询与「来源/方式」列幻觉**：① ontology 原来没有这两类动作，短语「让对方场上的怪兽的效果无效化」被 `"对方" in p and ("离开" in p or "场上" in p)` 误映射成 `leaves_field_by_opponent`；② `search_by_effect` 兜底路径只返回卡名、**不给片段**，LLM 只能自拟概述 → 编出原文没有的「融合素材以外」「陷阱种类2以上」（真实卡文是「那只怪兽以外的…」「2种类以上」）。→ 本体加 `negate_effect`（效果无效化/效果无效/无效化）与 `cannot_activate`（不能把…效果发动/不能发动/不可发动），map 接线（「无效」→negate、「不能」+「发动」→cannot_activate，leaves 必须含「离开/离场」）；否定句豁免只保留 `cannot_activate`（`对方若不从卡组送墓则不能把魔法卡发动` 里的送墓是自肃条件，不能算动作）；`snippet` 一律给**整段动作单元/整分句原文**（≤120 字，含「那只怪兽以外」这类限定语）；`search_by_effect` 兜底也补 `_matched_sentence`（按命中词权重选分句）；REACT_RULE 要求表格「来源/方式」列**照抄片段**，缺片段写「见卡牌原文」。坑：别把「无效/无效化」塞进 `VERB_WORDS`——「效果无效从卡组特殊召唤」里它是形容词，会截断来源区（赤舍利军贯 63748694 会从检索里消失）。
31. **同义译法被当成机制差异（AI 判「不包含」）**：用户问「无穷之三幻魔-幻魔皇 拉比艾尔 的①效果包含『舍弃手卡』吗」——原文「选自己1张手卡丢弃」，模型拿 P02 的「措辞严格」判「不包含」。→ 边界：**译法差异**（舍弃↔丢弃、手牌↔手卡、送入墓地↔送去墓地、取对象↔为对象）视为同一动作，回答注明原文措辞即可；**机制差异**（丢弃≠送去墓地≠作为cost、取对象≠发动后挑选、cost段≠效果处理）仍严格区分。已写进 `TASK_BODY.ruling #5` / `REACT_RULE #10`，并落判例 C29（`cases.jsonl`，must_cite 96345184）。注意检索层 SYNS/aliases 本来就归一（`search_by_action('舍弃手卡')` 能查到这张卡），问题只在问答层措辞判断。
32. **触发条件被误标成 cost**：查「从卡组送入墓地的 cost」时返回「自己卡组的卡被效果送去墓地的场合才能发动」「这张卡从手卡·卡组送去墓地的场合才能发动」——这些是**发动（触发）条件**，在「才能发动」之前但不是 cost（cost 是「把…超量素材取除」这种祈使动作）。→ `effect_parser` 给每个动作单元定 `reason`：①「场合/場合」**任意位置**出现即条件（不限于句尾——「把…解放的场合可以特殊召唤」是特殊召唤条件，不是 cost）；②「…时/…とき/…的时候/…之际」收尾也是条件，可带「(才能)发动」；③无「才能发动」的句式同样判定（缚神冢②「…的场合发动」、熔岩魔神特殊召唤条件）；④cost 段里以「把/将/从/以/解放/丢弃/支付/作为cost」开头=真 cost，含「被」的被动句=条件的一部分（猛吹雪「被…效果破坏，从场地送去墓地时才能发动」前半句不是 cost）。`search_by_action(reason='cost')` 只认 `reason=cost`（森罗的守神等条件卡退出，cost 命中 101→76），真 cost 保留。**已知限制**：「…的回合」这类自肃条件（熔岩魔神「把这张卡特殊召唤的回合，自己不能通常召唤」）不命中条件词，仍归 effect。
33. **cost/效果边界被模型读反（基路伯③）**：C21 曾判 NO——模型把「从卡组把1只3星怪兽送去墓地，以…为对象才能发动」的条款**顺序读反**，把 cost 当效果段。→ ① `agent/tools.py._labeled_snippet()` 给命中片段加段位前缀 `【cost】/【effect】/【trigger】/【旧措辞】`（parser 按「才能发动」切好的结论，表头有图例），模型直接采信不再猜；② REACT_RULE #11：以「才能发动」为界判 cost/效果，片段不含边界词时看完整原文、按条款原顺序判断，不得反推；③ `eval/run.py` 固定 `temperature=0.0`（默认 0.2 会引入采样方差）。**新动作**（本体 14→17 类）：attack_declare 攻击宣言、reveal 翻开/公开、set 盖放/覆盖，special_summon 扩到含「召唤」（覆盖通常召唤/召唤成功/反转召唤）。结果：C21 回归通过；C21/C23/C28 计数不变，C24 因「召唤」扩词 1160→1188。**长名单漂移（C23 35/24 vs 36/23）**：模型会自行重新分组/漏抄 → 已加 REACT_RULE #12（枚举结果原样复制工具分组与张数，不得重新分组/调换/增删）。
34. **禁限卡表三连 bug（limit_status 永远「无限制」）**：`ygocdb limits.json` 里 ①列表元素是**字符串**编号、②编号是 **cid**（内部编号）不是密码 id、③准限键名是 `semi_limited` 不是 `semiLimited`。原 `limit_status` 拿密码 id 直接 `in` 比对 → 恒「无限制」；概览准限恒 0。→ `client.get_limits()` 统一归一成 `{lang: {forbidden/limited/semi_limited: set[int], date}}`；`ToolRuntime._cid_of()` 用本地 cards 表把**密码 id→cid** 再比对（`limit_status` 兼容直接传 cid 的容错）；`cli limits` 同步改用归一键。验证：恶魔科学怪人/强欲之壶=禁卡（cn/ja/en）、超融合 cn/ja 准限制、朱光 ja 限制、灰流丽 ja 准限制。数据日期 cn/ja 2026-10-01、en 2026-10-19。注意数据源是**在线拉取**（运行期缓存，不落盘），网络抖动时会超时（曾 `WinError 10060`）。
35. **精确度两个坑（C21/C33 暴露）**：① 「3星以下 / 3星以上」被子串判断当成精确限定 3 星（`m in text`）→ 通用卡混进 ★精确组（C21 判 NO 的直接原因）；→ `tools._mod_exact()` 排除 以下/以上/以内/未满/超过 后缀，`search_by_action` 与 `search_by_effect` 两处都用它。② `_labeled_snippet()` 段位标签为 `None` 时 `None + " "` 崩溃（`search_by_effect` 结构化条目没有 reason 字段）→ 改 `(lab or "")`。另：C33 原设计为"触发条件检索题"，438 张的范围与 judge 框架反复冲突（时点/主体两分类），已改为"『…攻击宣言时』是发动条件还是 cost"的判断题（稳定、且测的正是 [trigger] 识别）；翻开类检索由 C34 覆盖（已改为"翻开 vs 送墓"判断题）。
36. **禁限状态回答与「未找到该卡」弹窗**：① 模型把「超融合」日文表答成「无限制」并**编造韩文「无限制」**——工具 `limit_status` 原本对不支持语言 `lang=ko` 静默返回「无限制」（`d={}` 落空），且提示词没限制语言范围。→ `limit_status` 校验语言（仅 cn/ja/en，日文别名 jp→ja；其它明确返回「不支持的语言」），TOOLS schema 的 `lang` 加 `enum`，REACT_RULE #14 禁止增加不存在的语言/按常识推断状态。验证：超融合 cn=准限制、ja=准限制、en=无限制；`ko` 返回不支持。② `_cid_of()` 曾 `db.close()` 关掉 **store 共享连接**（`connect()` 返回同一 `_db`）→ 之后所有 DB 操作报 "Cannot operate on a closed database"，表现为卡牌详情/搜索 500/404。→ 移除 close，复用共享连接。③ 答案里引用的是 **cid** 时 `/api/card/<cid>` 404 → 前端 `alert("未找到该卡")` 弹窗；→ `api_card` 增加 `get_by_cid` 回退，前端把 alert 换成**非阻塞 toast**（`.toast` 样式，3 秒淡出），不再打断操作。

## 扩展模式
- **加工具**：`ToolRuntime` 加方法（返回 `(text, trace)`）→ `agent._prepare` 按意图路由调用 → `trace.tools.append(...)` →（可选）`TOOLS` JSON schema 供原生 function-calling →（可选）`web/server.py` 加 `/api/xxx` 端点 + `app.js` 调用。
- **加切分器**：`rag/chunker.py` 加类（输出 `Chunk(text, meta)`）→ `CHUNKERS` 注册 → `build_index`/`build_faq_index` 用。
- **加原则**：`agent/prompts.py` 的 `PRINCIPLES_BLOCK` 加条目（ruling 系统提示自动含）。
- **加人设**：`PERSONAS` 字典加 → `build_system(task, persona)` 自动支持。
- **加推理参数**：全链路透传——`agent.ask/ask_stream/_prepare` → `budget.fit` → `build_messages`（参考 persona/history/max_context/thinking 的透传）。
- **新搜索维度**：参考 `search_by_effect`（PSCT 解析+同义词+主动/被动+全量两档）或 `search_by_series`（LCS 提取 token+name LIKE 全量）。

## 编码约定
- **零依赖优先**：标准库能做就不引包；额外依赖放 `[extras]`（llm/web/embed），mock 后端兜底让无 Key 也能跑。
- **对上层透明**：`Embedder`/`LLMClient` 暴露统一方法，换后端只改构造一行。
- **trace 全程可见**：工具链/预算/token 用量都进 trace，可复盘。
- **尊重数据源**：cards.zip 走 MD5 增量、FAQ 磁盘缓存、`YgocdbClient(min_interval=0.3)` 限流。
- **渐进披露**：本 skill 精简，深度内容查 `README.md` 与 `docs/技术路线.md`（技术路线 / 取舍记录）。
- **裁定领域知识**（排查裁定答案对错时）：cost≠效果、取对象≠发动后挑选、触发字面性（丢弃≠送去墓地≠cost）、90X 后 PSCT 统一但"未统一≠裁定不同"（判措辞非日期）、原则+FAQ 双轨。原则原文在 `agent/prompts.py`。

## 边界
- 只做 AB 层（推理参数 + 上下文工程），不做 C 层权重微调（用闭源 API）。
- 动作/原因类细分（"因效果送入墓地"）无结构化字段，靠 FAQ+原则；身份类（种族/类型/星级）位掩码精确。
- mock 后端不支持原生 tool-calling → mock 走确定性管线兜底；真模型走 ReAct 多轮（trace 结构一致，同为 `to_event()`）。
