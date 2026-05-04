# FinDoc Agent 学习笔记

## 一、项目定位

面向**中文金融文档（年报、招股书、研报）的多模态 RAG Agent**。核心思路：绕过 OCR + 文本分块，直接用**视觉检索**定位页面，再用 VLM 读取内容，最后由 LLM 合成带引用的答案。

---

## 二、整体工作流

```
用户问题
    │
    ▼
┌──────────┐     拆解为有序子任务（JSON plan）
│ Planner  │ ──► 每个 SubTask: query / target_doc / expected_schema
└──────────┘
    │
    ▼
┌──────────┐     text/table → ColQwen 检索 → VLM 读页 → 抽取事实
│ Executor │ ──►
└──────────┘     number     → calculator AST 安全求值
    │
    ▼
┌──────────┐     ✅ 充分 → Synthesizer
│ Verifier │ ──►
└──────────┘     ↻ 不充分 → 追加 sub_task → Executor（≤3 轮）
    │
    ▼
┌─────────────┐
│ Synthesizer │  汇总事实 + 计算值 → 带 [doc_id p.X] 引用的最终答案
└─────────────┘
```

---

## 三、四个 Agent 节点

| 节点 | 职责 |
|---|---|
| **Planner** | DeepSeek 拆解问题为有序 SubTask，标注期望输出（text/number/table） |
| **Executor** | 按 schema 路由：number → calculator；其他 → ColQwen 检索 + VLM 读页 |
| **Verifier** | 双重校验：充分性（证据是否够答）+ 一致性（跨页数字是否对得上）。不充分时追加 SubTask 重入 Executor，超 3 轮强制进 Synthesizer |
| **Synthesizer** | 汇总事实生成最终答案，每条事实必须附 `[doc_id p.X]` 引用，不引入证据外信息 |

---

## 四、三类模型，各司其职

| 模型 | 调用方式 | 用途 |
|---|---|---|
| **DeepSeek V4 Flash** | OpenAI-Compatible API | Planner / Verifier / Synthesizer 推理（纯文本） |
| **ColQwen2** | 本地 GPU（Litserve 服务化） | 视觉检索 backbone，多向量 + MaxSim |
| **Qwen VLM** | DashScope API | 页面图 → 结构化文本 |

### 为什么 ColQwen 是核心

传统 RAG：PDF → OCR → 文本分块 → 单向量 embedding → 语义搜索。金融文档表格密集、版式复杂，OCR 出错、分块破坏表格语义。

ColQwen 直接用视觉编码器：
- 每页图片 → 多向量 embedding（每个 patch 一个向量）
- 查询同样编码为多向量
- **MaxSim**（query token × page token 取最大相似度，再求和）算相关性
- 表格、图表、数字布局信息完整保留

### 协作流程

```
"茅台2023年毛利率"
      │
      ▼
   ColQwen2 ──► 多向量 query
      │
      │ MaxSim 匹配（in_memory 或 Qdrant 服务端）
      ▼
   top-k 页面图片
      │
      │ 图片 + 指令
      ▼
   Qwen VLM ──► "毛利率94.12%"
      │
      ▼
   DeepSeek ──► 带引用最终答案
```

ColQwen = **检索引擎**（找哪几页有答案），Qwen VLM = **阅读器**（从图里读内容），DeepSeek = **推理大脑**（规划/验证/合成）。

---

## 五、当前架构（P10 之后）

```
┌─ uvicorn backend.server:app (port 8001) ──────┐
│  FastAPI + SSE                                  │
│  POST /api/v1/query  → SSE 流（节点 + 进度）    │
│  GET  /api/v1/docs   → 已索引文档列表           │
│  GET  /api/v1/health                            │
│  startup → 预加载 ColQwen2 + 索引               │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴────────────┐
        ▼                       ▼
┌────────────────┐    ┌──────────────────────┐
│ ColQwen Service│    │ Qdrant (多向量+MaxSim)│
│ Litserve+GPU   │    │ Docker · port 6333    │
│ port 8000      │    │ collection: findoc_pages│
└────────────────┘    └──────────────────────┘

┌─ chainlit run app/chainlit_app.py ────────────┐
│  纯 UI 层（仅 import chainlit / httpx）         │
│  消费 SSE：event=node|status|done|error         │
│  动态 Step + 引用 inline Image                  │
└──────────────────────────────────────────────────┘
```

**关键边界**：`agent/` `tools/` `ingestion/` 三个目录是业务核心，被后端直接 import；前端通过 HTTP/SSE 调用，不再 import 任何 agent 代码。

---

## 六、核心技术点速查

### MaxSim（多向量后期交互）

```
score(q, d) = Σ over q_tokens : max over d_tokens : ⟨q_i, d_j⟩
```

- Python 实现：`einsum("td,pkd->ptk", q, d).max(dim=-1).sum(dim=-1)`
- Qdrant 实现：`MultiVectorConfig(comparator=MAX_SIM)` + `Distance.DOT`（不归一化，与 Python 一致；用 COSINE 会先归一化导致结果不同）
- 一份 200 页年报 ≈ 200K 向量（每页 ~1024 patch × 128 维）
- 单向量库（Chroma/pgvector）无法支持，只有 Qdrant 1.10+/Milvus 2.5+/Vespa 原生支持

### Reflexion 循环

参考 Shinn et al., 2023，但限定为"客观可判定"两类校验（充分性 + 一致性），不做通用 self-critique（容易让 LLM 自我说服）。`max_reflexion_iter=3` 防死循环，超出标低置信。

### SSE 进度推送（P10）

LangGraph `astream` 在 Executor 内部被模型加载/VLM 调用阻塞，无法从主循环推中间事件。解法：

- 工具侧（colpali_tool / vlm_tool）：模块级 `_progress_hook` 回调 + `_report_progress(msg)` 在长操作点埋点
- 后端侧：Agent 放后台线程，主 async 循环每 300ms 轮询节点队列 + 进度队列，交替 yield `event: node` 和 `event: status`
- 前端：进度 Step 名字随消息变化（`⏳ VLM 正在读取 p009.png...`），完成切 `✅ 完成`

侵入小（每个工具 ~15 行），不改任何函数签名。

---

## 七、四种部署组合（config 矩阵）

| `services.colqwen_url` | `retriever.backend` | 含义 |
|---|---|---|
| `""` | `in_memory` | 默认开发模式：本地模型 + 本地 .pt MaxSim |
| `http://localhost:8000` | `in_memory` | ColQwen 服务化 + 本地 MaxSim |
| `""` | `qdrant` | 本地编码 + Qdrant 服务端 MaxSim |
| `http://localhost:8000` | `qdrant` | 完全分布式（推荐生产） |

任何 Qdrant 异常自动 fallback 到 `_in_memory_retrieve`，不让 agent 崩溃。

---

## 八、剩余的工业级差距

P7/P8/P10 已解决了"架构解耦 / 横向扩展 / 可观测性入口 / 多用户竞态"。下面是仍未补的硬伤：

| 项 | 现状 | 建议 |
|---|---|---|
| **测试** | 零测试文件 | `tests/` + pytest，先覆盖 calculator AST 安全求值、`derive_doc_id`、MaxSim 确定性张量 |
| **缓存** | 无 | VLM 页面读取按 `(doc_id, page_num, query_hash)` 落 SQLite；LLM 输出按 query+plan 哈希做语义缓存 |
| **重试与超时** | 失败直接 fallback 到 stub | 所有外部 API 加 `tenacity` 指数退避；ChatOpenAI 加 `timeout=60, max_retries=3` |
| **可观测性** | loguru 导入未配置 | 接 LangSmith（`LANGCHAIN_TRACING_V2=true`）；`logger.add(rotation, retention, serialize=True)` |
| **VLM 并行化** | top_k=5 串行调用 ~10s | `asyncio.gather` 并发 → ~2s |
| **持久化** | Chainlit 历史纯内存 | SQLite + aiosqlite 接 Chainlit Data Layer |
| **鉴权** | 无 | Chainlit `@cl.password_auth_callback` 或 OAuth |
| **CI/CD** | 无 | Dockerfile + docker-compose + GitHub Actions（ruff + pytest） |
| **配置校验** | 启动不检查必需 env | `pydantic-settings` 启动时校验所有 key |

---

## 十、Agent 核心调用逻辑深度分析（as-is 现状 + 设计弱点）

> 项目核心是 agent 设计，不是 RAG / 不是 UI。这一节把当前 4 节点 + 3 工具的实际行为拆开看，标出设计弱点和升级方向。后续 P19+ 路线图（见 DEVLOG）按这里的诊断展开。

整体调用流：`planner → executor →（per sub-task）retrieval+VLM 或 calculator → verifier →（不充分）回 executor / 充分 → synthesizer`。下面按 5 个轴线分析。

---

### 10.1 Planner — 「盲规划」是最大的结构性弱点

**As-is**（`agent/nodes/planner.py`）：

- 一次 LLM 调用，结构化输出 `PlannerOutput { plan: [PlanItemSchema] }`
- 输入：`query` + `doc_metadata`（仅 `doc_id, page_count` 列表）+ `chat_history`
- 输出：有序 `SubTask(sub_query, target_doc?, expected_output_schema)`
- prompt 限制：`number` schema 必须是纯算术，否则用 `text`
- 失败回退：单条 SubTask = 整个原 query

**设计弱点**：

1. **盲规划 / 无检索反馈**：planner 完全看不到候选页的内容，凭 doc_id 字面（如 `moutai_2023`）猜该不该 target 它。用户上传的 doc_id 是 uuid 后缀（`balance_sheet_a3f9c1`），planner 几乎只能填 `target_doc=null`。
2. **`doc_metadata` 太浅**：只有 doc_id + 页数。没有"这份文档讲什么主题、覆盖什么年份、有什么核心表格"的提示。LLM 凭 doc_id 字面 + chat_history 拼凑——错配率高。
3. **两阶段算式模式实际死代码**：prompt 鼓励"先 text 取数 → 再 number 算式"，但要求 planner 在规划期已经把字面数字写进表达式。它在规划时还没读到任何页，根本没法填——所以这条路径几乎不触发，绝大多数运算转到 synthesizer 写 prose 里手算。
4. **Plan 一次成型，零修订**：plan 一旦提交，剩余唯一修改路径是 verifier 在末尾 append 一条。如果 sub_task 1 跑完发现"该公司 2022 年是合并口径变更年"，sub_task 2/3 还在按旧口径走，浪费 LLM + VLM。
5. **子任务串行**：executor 用 `plan_cursor++` 一条一条跑。两个独立的"查 A 公司营收"和"查 B 公司营收"也只能串行——明显的并发空缺。
6. **没有"重要性 / 可选"标签**：所有 sub_task 等权。没机制让 planner 说"这步是核心、这步是探索性"，verifier 也无法据此分配 reflexion 预算。

**升级方向**：

- **检索感知规划（pre-retrieve）**：planner 之前先粗召回（query → ColQwen 全库 top-3 文档），把 candidate docs + 页数 + 第一页 caption 作为 metadata 喂进 planner prompt。
- **结构化决策模板**：内置「比较类 / 趋势类 / 计算类 / 单事实类」四种模板，planner 先分类再实例化模板，减少自由发挥的错配。
- **Plan 修订节点**：每个 sub_task 完成后插一个 `plan_critic`，根据新事实判断剩余 plan 是否还合理；不合理则触发 revise（成本：每问多 0–2 次轻量 LLM）。
- **独立 sub_task 并发**：依赖图（DAG）替代线性 cursor，独立步骤 `asyncio.gather`。

---

### 10.2 Tool 使用 — 路由是写死的 if/else，工具不可发现

**As-is**（`agent/nodes/executor.py`）：

```python
if sub_task.expected_output_schema == "number":
    return _run_calculation(sub_task, cursor)   # → calculator.calculate(expr)
return _run_retrieval(sub_task, cursor, ...)     # → colpali_retrieve + vlm_read_page × top_k
```

工具集：`colpali_retrieve` / `vlm_read_page` / `calculate`。

**设计弱点**：

1. **Dispatch 写死**：靠 `expected_output_schema` 字符串硬路由。加一个新工具（SQL、网页抓取、表格抽取）就得改 executor + planner prompt + schemas，改面太大。这不是真正的 "agent 调工具"，是"两条预设流水线"。
2. **工具没有自描述**：现代 tool-calling agent 标配 `(name, description, JSON params schema)` 让 LLM 自选；这里 planner 不知道"我有哪些工具可用、它们要什么输入、返回什么"，只能套路 retrieval+VLM。
3. **VLM 是粗浪费的**：每次 `_run_retrieval` 把 top-k=5 页**全部**发给 VLM，每页用相同的 `instruction = sub_task.sub_query`。命中率高的那 1 页其实就够；剩下 4 次 VLM call（每次 1–2s）几乎纯浪费。已通过 ThreadPool 改成并发，但**总调用量**没降。
4. **没有相关性阈值短路**：ColQwen 给 score 后没人用，所有 top-k 一视同仁送 VLM。一份完全不相关的文档（score 远低于均值）也照样走完。
5. **没 rerank 二阶段**：ColQwen 多向量召回宽而粗；缺一层廉价 cross-encoder 对 (query, page-caption) 做精排，把 top-20 收敛到 top-3。
6. **VLM 输出是 free text，工具不强制 schema**：sub_task 说"我要 number"，VLM 仍可能返回散文。executor 不解析校验，把整段塞进 `Fact.text`，后面 verifier / synthesizer 各自再自己解析。
7. **VLM 指令太粗**：`instruction = sub_task.sub_query` 直接用。"茅台2023营收"丢给 VLM，它可能给整页摘要而不是定向取数。应该有 instruction 模板（"在本页查找 X，仅返回数值 + 单位 + 行标签；找不到回 not_on_page"）。
8. **没有 tool-call 的统一日志**：每个工具自己 logger.warning + progress hook。没有"本轮 plan 用了哪些工具、各用了几次、平均耗时"的横向视图。

**升级方向**：

- **Tool registry + JSON schema dispatch**：每个工具注册 `{name, description, params_schema, callable}`；planner 输出 `tool_calls: [{tool, args}]` 替代 schema 字符串。Executor 仅做 dispatch + 校验。
- **检索 → rerank → VLM 三阶段**：top-20 → bge-reranker top-3 → VLM。
- **VLM 指令模板化**：按 `expected_output_schema` 选 prompt（数值取数 / 表格抽取 / 段落概述）。
- **早停**：VLM 返回 `not_on_page` / 置信度低时跳过本页；有一页给出完整答案后剩余可短路（可选）。

---

### 10.3 Reflexion — `missing_info` 是字符串，丢失了根因

**As-is**（`agent/nodes/verifier.py`）：

```python
result: VerifierOutput = llm.invoke(prompt)  # is_sufficient / inconsistency / missing_info
if not sufficient:
    new_plan = list(plan) + [SubTask(sub_query=missing_info)]  # 只有 sub_query，没 target_doc / 没 schema
```

`max_reflexion_iter = 3`，超出强制走 synthesizer。

**设计弱点**：

1. **`missing_info: str` 把诊断信息全压平**：verifier 实际能判断"是哪一步缺、缺什么、应该用什么 doc"，但被压成一行自由文本，executor 重跑时只看到 sub_query，target_doc / schema 全丢。
2. **不区分根因**：补救手段被绑死成"再来一次 retrieval+VLM"。但失败原因可能是：
   - retrieval 没召回正确页（应：放宽 top_k / 改写 query）
   - retrieval 召对了 VLM 读错了（应：换更精细 instruction 重读同一页）
   - 原 sub_query 本身歧义（应：planner 改写）
   - 跨页数字不一致（应：调取报表披露口径页 / 用 ranker 决谁权威）
   现在四种全用同一种"再 retrieve+VLM"补救，浪费且经常修不上。
3. **没有 reflexion 记忆**：第 1 轮 reflexion 跑了 query "X" 没结果；第 2 轮 verifier 完全可能再发 query "X"——没记 `tried_queries` 集合。
4. **不一致时的修复弱**：`inconsistency` 非空就强制 `is_sufficient=false` 并把 inconsistency 字符串塞进 missing_info。但跨页数字冲突真正需要的是"挑出权威页 / 解释为何不同口径"，不是再 retrieve。
5. **二元 sufficiency**：`is_sufficient: bool` 没置信度。"勉强够"和"明确够"得到同等待遇，对高风险数值题不够。
6. **verifier 看不到 chat_history**：用户的指代消解只在 planner 端做。验证阶段如果对"它/上述"产生歧义，verifier 没历史可参考。
7. **3 轮硬上限缺软信号**：一轮跑完没拿到任何**新事实**也不会提早停——继续浪费。

**升级方向**：

- **结构化 missing_facts**：每条带 `{sub_task_idx, what, root_cause: retrieval|reading|ambiguous|inconsistency, suggested_remedy: re_retrieve|re_read|rewrite|disambiguate, suggested_query?, suggested_target_doc?}`
- **按 root_cause 分派修复策略**（详见 P20 计划）
- **reflexion 状态记录**：在 state 里加 `tried_queries: list[str]`，verifier prompt 看到后避免重复
- **早停信号**：本轮 reflexion 后 `extracted_facts` 数量未增加 → 直接强制走 synthesizer（标 low-confidence）

---

### 10.4 Memory — chat_history 只到 planner，事实 / 实体没沉淀

**As-is**：

- `chat_history`：SQLite 拉最近 4 轮 (user, assistant)，**只**注入 planner prompt
- `extracted_facts`：单 query 内累积，跨 query **不复用**
- VLM cache：`(image_path, instruction)` → 文本（done）
- `colpali_retrieve` 没 cache：相同 query 重跑也重算

**设计弱点**：

1. **memory 只服务 planner**：verifier 和 synthesizer 看不到 history。问"它的同期增长率呢"，synthesizer 不知道"它"是哪家公司，可能给出错误指代或拒答。
2. **history 是消息文本，不是结构化事实**：传 `{role, content}` 让 LLM 每次重新解析"上一轮提到了茅台"。token 浪费 + 解析不稳。应该提取 entity slot：`last_focus_doc=moutai_2023, last_metric=毛利率, last_value=91.52%`。
3. **跨轮事实零复用**：第 1 问拿到"茅台 2023 营收 1505.6 亿"；第 2 问"那 2022 年呢"——重跑 retrieval + VLM。理想：把上轮 fact 持久化（`(doc_id, page_num, claim_type, value, ts)`），下轮 query 先查 fact store。
4. **单 query 内的事实是扁平 list**：`extracted_facts: list[Fact]` 是平的。planner 后续步骤拿不到结构化的"已知 X、未知 Y"——verifier 只能整体判断。
5. **没有"用户 / 文档 profile"长期记忆**：高频被问的实体（用户老问茅台）没在检索阶段倾斜。
6. **retrieval cache 缺失**：相同 query 反复跑（用户用 starter 重复同问、benchmark 重复跑）每次都走完整 ColQwen 编码 + MaxSim。

**升级方向**：

- **多节点 memory 注入**：把 history（或 history 摘要）也注入 verifier / synthesizer prompt
- **从 history 抽 entity slots**：用一次轻量 LLM 把历史压成 `last_entities`，比传消息文本省 token 又准
- **跨轮 fact store**：`facts` 表 `(conv_id, doc_id, page_num, claim_type, value)`；plan 之前先查命中
- **fact store 结构化**：`Fact.text` 升级到 `{kind: numeric|string|table_row, key, value, unit, period}`，verifier 可机械判断"是否已有 (公司=茅台, 期间=2023, 指标=营收)"
- **retrieval 语义缓存**：query embedding 哈希为 key

---

### 10.5 Verification — 验的是「够不够」，不验「答案是否真贴合证据」

**As-is**：

- verifier 节点：判 sufficiency + inconsistency
- synthesizer 节点直接输出最终回答
- **没有任何后置校验**：synthesizer 写的内容只要 LLM 返回就直接给用户

**设计弱点**：

1. **synthesizer 是个 trust-blackbox**：verifier 在合成**之前**判证据够不够；合成之后没人看产物。LLM 完全可以幻觉一个引用 `[fake_doc p.99]`，或把 1505.6 亿写成 1605.6 亿。
2. **citation 真实性没校验**：synthesizer prompt 要求"每条事实必须 `[doc_id p.X]` 引用"，但有没有真引用、引用是否在 evidence 里出现过——零校验。
3. **数值一致性没校验**：synthesizer 写 "营收 1505.6 亿元"，跟 `extracted_facts` 里的原数据是否字面对得上——无人检查。VLM 出错或 LLM 复述时改写都不会被发现。
4. **单位 / 口径校验只在 prompt 里口头说**：verifier prompt 提到要查"合并 vs 母公司、FY vs CY"，但没工具支撑——LLM 自己判断，准确性没保证。
5. **单 verifier 单判**：高风险数值题没有 ensemble / self-consistency。
6. **没有 fail-safe 输出格式**：当 verifier 标 low-confidence 时，前端只看到答案，不知道是"高置信"还是"凑出来的"；UI 没有醒目降级标识。

**升级方向**：

- **Citation 反向校验节点**：synthesizer 之后插一个 lightweight 节点，正则提取 `[doc p.X]`，逐一查 `extracted_facts`；不存在的引用标 unverified 并从最终输出剥离。
- **Numeric 校验**：synthesizer 输出里所有数字都尝试 match 回 `extracted_facts.text`；阈值容差内才放行。
- **置信度显式化**：verifier 输出 `confidence: float`；synthesizer 在低置信下加 banner "⚠ 信息有限/口径不明"。
- **可选：self-consistency verifier**：3 次采样 verifier 输出多数表决（高风险数值题用，平时不用，控制成本）。

---

### 10.6 总结：核心调用逻辑的 5 处优先升级点

按"改了一处带动多处"排序：

1. **Verifier 结构化 missing_facts + root_cause**——影响整个 reflexion 链路（P19）
2. **按 root_cause 差异化修复策略**——把 reflexion 从"再来一次"升级到"对症下药"（P20）
3. **Tool registry + JSON schema dispatch**——把 executor 从两条硬流水线改成可扩展架构（P21）
4. **检索感知规划（pre-retrieve candidate docs）**——把 planner 从"盲规划"升级（P22）
5. **后置 grounding 校验（citation + numeric）**——补上"答案对不对"的最后一环（P23）

---

## 九、亮点（值得保留的设计）

- **视觉检索绕过 OCR**：ColPali/ColQwen 是 2024 年才兴起的方向，多数 RAG 项目仍走 OCR + 文本分块
- **Planner-Verifier-Synthesizer 显式状态机**：不是简单 QA 链，有真正的 reflexion 循环
- **MaxSim 后期交互**：比 CLIP/ViT 单向量检索强一档
- **金融领域适配**：calculator AST 求值、跨页一致性校验、引用溯源
- **三层服务边界清晰**：ColQwen Service（GPU）/ Qdrant（向量库）/ FastAPI（业务）/ Chainlit（UI），任何一层可独立替换
