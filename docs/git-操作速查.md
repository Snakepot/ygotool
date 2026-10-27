# Git 操作速查（本仓库日常用）

> 本仓库已建立本地 git，第一次提交是 `v0.1.0` 基线。
> 核心心法：**改大东西前先 `git add -A && git commit` 存一份现场；改坏了随时用 `git restore` 找回。**
> 注意：`.env`（含 API Key）和 `cache/`（卡池、索引，可重建）**不入库**，别觉得它们没被管就是丢了。

## 一、每次改完代码（最常见的操作）

```powershell
git status              # 看哪些文件被改了（红色=未存档，绿色=已暂存）
git add -A              # 把所有改动放进"暂存区"（准备存档）
git commit -m "描述一下改了什么，例如：修复 Web 检索后端不匹配"
```

## 二、发现改坏了，想找回

```powershell
# 1) 只恢复某个文件到上次提交的样子（未提交的改动会丢）
git restore web/server.py

# 2) 恢复全部未提交改动（回到上次 commit 的干净状态）
git restore .

# 3) 改动已经 commit 了，想回退到某个版本（历史保留，安全）：
git log --oneline          # 看历史，找到目标版本号或标签
git reset --hard v0.1.0    # 整个工作区回到 v0.1.0（会直接覆盖磁盘，确认不要当前改动再用）

# 4) 已经 commit 想"撤销提交但保留文件改动"：
git reset --soft HEAD~1    # 撤销最近一次 commit，改动退回暂存区，文件内容不变
```

## 三、只是想看一眼旧代码（不丢任何东西）

```powershell
git log --oneline                       # 列出历史
git show <版本号>:web/server.py         # 打印某版本下某文件的内容
git checkout v0.1.0 -- web/server.py    # 把某版本下某文件取回到工作区（之后记得 commit）
```

## 四、建议的节奏（给"怕改坏"的安全垫）

1. **动手前先存现场**：`git add -A; git commit -m "存档：改 XX 前"`
2. 改完能跑通，再 `git add -A; git commit -m "功能：XX"`
3. 大改动可建分支，玩坏了直接删分支：`git switch -c try-xxx`；失败后 `git switch main` 并 `git branch -D try-xxx`
4. 想更保险可随时打标签：`git tag 改前`
5. 想看看今天改了哪些文件：`git status --short`

## 五、小抄

| 场景 | 命令 |
|---|---|
| 看改了什么 | `git status` / `git diff` / `git diff --stat` |
| 存档 | `git add -A && git commit -m "说明"` |
| 撤销未提交的单文件 | `git restore 文件名` |
| 撤销全部未提交 | `git restore .` |
| 回到某次提交（覆盖磁盘） | `git reset --hard <版本号/标签>` |
| 撤销最近一次 commit（保留改动） | `git reset --soft HEAD~1` |
| 历史列表 | `git log --oneline` |
| 给当前存档起名 | `git tag 名称` |

## 六、要小心的事

- `git reset --hard` 会**覆盖磁盘**，只在你确认"这些改动不要了"时用；
- `.env` 不在版本控制里，改配置前自己另存一份（或把配置项写进 `.env.example`，在 `.env` 填值）；
- `cache/` 同理：卡池和索引用 `python cli.py refresh` / `build-index` 重新生成；
- 不要把 `.env` 里的 Key 贴到 commit 消息、issue、聊天里。