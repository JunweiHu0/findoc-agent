# FinDoc Agent 学习笔记

## 一、项目定位

面向**中文金融文档（年报、招股书、研报）的多模态 RAG Agent**。核心思路：绕过 OCR + 文本分块，直接用**视觉检索**定位页面，再用 VLM 读取内容，最后由 LLM 合成带引用的答案。

---

## 二、当前架构（P26–P32 后，8 节点）

```
retrieval_scout → planner → executor → plan_critic? ──┐
                    ▲          ▲      │   (信号触发)     │
                    │          │      ▼                 ▼
                    │          │    executor          verifier ──┐
                    │          │                        │ sufficient?
                    │          └── remediation ◄────────┤ not
                    │                                   ▼ sufficient
                    └── plan revision ──── plan_critic ◄── (signal)
                                                        │
                                              synthesizer → grounding → END
```

| 节点 | 职责 | 来源 |
|------|------|------|
| **retrieval_scout** | 预检索全库 top-3 文档，让 planner 知道哪些 doc 可能相关 | P22 |
| **planner** | 两段式：query_class 分类 → variant prompt + few-shot 生成 plan（含 DAG 依赖） | P30 |
| **executor** | DAG 拓扑调度，同层无依赖 task 并发执行（ThreadPool max 5） | P29 |
| **plan_critic** | 信号词（口径变更/任务失败）触发，~200 token LLM 判断剩余 plan 是否需修订 | P29 |
| **verifier** | 计算/对比类启用 3 实例并行多数表决（strict/base/numeric）；其他单实例 | P19/P31 |
| **remediation** | 按 root_cause 分派 4 条修复路径，含 budget 控制 + retry TodoItem 追踪 | P20/P26.5 |
| **synthesizer** | 按 query_class 选 variant prompt，流式 token-by-token 输出 | P16/P30 |
| **grounding** | Citation 反向校验 + numeric fuzzy match ±0.1%，剥离未验证引用，回写事实验证状态 | P23/P28 |

### 三类模型

| 模型 | 调用方式 | 用途 |
|---|---|---|
| **DeepSeek V4 Flash** | OpenAI-Compatible API | Planner / Verifier / Synthesizer 推理 |
| **ColQwen2** | 本地 GPU / Litserve 服务化 | 视觉检索 backbone，多向量 + MaxSim |
| **Qwen VLM** | DashScope API | 页面图 → 结构化文本 |

**为什么 ColQwen 是核心**：传统 RAG `PDF → OCR → 文本分块 → 单向量 embedding` 破坏表格语义。ColQwen 直接用视觉编码器：每页图片 → 多向量（每 patch 一个向量），MaxSim 计算 `Σ max ⟨q_i, d_j⟩`，表格/图表/数字布局完整保留。

协作流程：`ColQwen（检索引擎）→ Qwen VLM（阅读器）→ DeepSeek（推理大脑）`

### 部署架构

```
┌─ FastAPI backend (port 8001) ───────────────────┐
│  SSE stream: event=node|status|token|todo|done   │
│  startup → 预加载 ColQwen2 + 索引                 │
└──────────────┬───────────────────────────────────┘
               │
    ┌──────────┴────────────┐
    ▼                       ▼
┌────────────┐    ┌──────────────────────┐
│ ColQwen    │    │ Qdrant (多向量+MaxSim)│
│ Litserve   │    │ Docker · port 6333    │
│ GPU :8000  │    │ findoc_pages          │
└────────────┘    └──────────────────────┘

┌─ Chainlit frontend ──────────────────────────────┐
│  纯 UI 层，通过 HTTP/SSE 调用后端                   │
│  动态 Step + 引用 inline Image + 置信度 banner      │
└──────────────────────────────────────────────────┘
```

---

## 三、核心技术点

### MaxSim（多向量后期交互）

```
score(q, d) = Σ over q_tokens : max over d_tokens : ⟨q_i, d_j⟩
```

- Python：`einsum("td,pkd->ptk", q, d).max(dim=-1).sum(dim=-1)`
- Qdrant：`MultiVectorConfig(comparator=MAX_SIM)` + `Distance.DOT`（不归一化，与 Python 一致）
- 一份 200 页年报 ≈ 200K 向量（每页 ~1024 patch × 128 维），单向量库无法支持

### Reflexion 循环

参考 Shinn et al., 2023，限定为"客观可判定"两类校验（充分性 + 一致性），不做通用 self-critique。`max_reflexion_iter=3` + `budget_retrievals` + `budget_vlm_calls` 三重保护。

### SSE 进度推送（P10）

工具侧 `_progress_hook` 回调在长操作点埋点 → 后端主 async 循环轮询队列交替 yield `event: node` 和 `event: status` → 前端实时更新 Step 名称。侵入小（每个工具 ~15 行）。

### 四种部署组合

| `services.colqwen_url` | `retriever.backend` | 含义 |
|---|---|---|
| `""` | `in_memory` | 开发模式：本地模型 + 本地 MaxSim |
| `http://localhost:8000` | `in_memory` | ColQwen 服务化 + 本地 MaxSim |
| `""` | `qdrant` | 本地编码 + Qdrant 服务端 MaxSim |
| `http://localhost:8000` | `qdrant` | 完全分布式（推荐生产） |

Qdrant 异常自动 fallback 到 `_in_memory_retrieve`。

---

## 四、架构演进：4 节点 → 7 节点 → 8 节点

### 4.1 最初版本（P1–P18）：4 节点直筒

```
planner → executor → verifier ⇄ executor → synthesizer
```

核心弱点（详见 DEVLOG §10）：

| # | 弱点 | 影响 |
|---|---|---|
| 1 | Planner "盲规划"：看不到候选页内容，凭 doc_id 字面猜 target | 错配率高 |
| 2 | Executor 路由写死 if/else，工具不可发现 | 加工具改面太大 |
| 3 | Verifier `missing_info: str` 丢失根因，四种失败全用同一种补救 | reflexion 浪费 |
| 4 | chat_history 只到 planner，跨轮事实零复用 | 追问重新检索 |
| 5 | Synthesizer 输出无后置校验，引用/数字可能幻觉 | 答案不可信 |

### 4.2 P19–P25：7 节点 + 结构化升级

新增 3 节点：`retrieval_scout`（解决盲规划）、`remediation`（按 root_cause 分派 4 条修复路径）、`grounding`（citation + numeric 后置校验）。

核心变化：
- Verifier 从 `missing_info: str` → 结构化 `MissingFact {root_cause, suggested_query, suggested_target_doc, ...}`
- Executor 从 `if schema == "number"` → Tool Registry（`retrieve_pages/read_page_with_vlm/calculate/disambiguate_caliber`，各带 params_schema + output_schema）
- Fact 从 `{text}` → `{entity, period, metric, value, unit, raw_kind}`，fact_extractor 纯 regex 结构化
- 跨轮事实复用：`conv_facts` 表 + `known_facts`，追问场景检索降 40–60%

### 4.3 P26–P32：8 节点 + 工程化深水区

新增 1 节点（`plan_critic`）+ 6 条系统能力线。按 Phase 组织：

**Phase 1 — 地基层（P26/P26.5）**：
- `agent/retry.py`：`with_retry` 装饰器，tenacity 指数退避（1s→2s→4s，上限 30s）。区分 transient（httpx.Timeout/429/5xx/openai.APITimeout/RateLimit）和 fatal（401/400，不重试）
- `llm.py`：`ChatOpenAI(timeout=60, max_retries=3)`
- `vlm_tool.py`/`colpali_tool.py`：API 调用加 `@with_retry`，VLM 仅重试 transient（坏图不反复消耗额度）
- `error_log: Annotated[list[dict], add]`：所有节点可追加结构化错误 `{error_type, message, retryable, fatal}`
- `TodoItem` + `todo_items`/`todo_updates`：运行时任务状态机 `pending→running→done|failed|skipped`，SSE `event: todo`，`turn_todos` 表持久化
- remediation retry 时 `attempt++`，`parent_id` 指向上次尝试，保留完整执行链路

**Phase 2 — 信息层（P27/P28）**：
- `agent/compression.py`：`compress_history`（>3 turns 用正则提取 `{entities, metrics, decisions, focus_doc}` 结构化摘要，只保留最近 2 turns 原文）；`compress_evidence`（按 `(entity,period,metric)` 去重保留最长 text）；`TokenBudget`（planner=1200/verifier=1500/synthesizer=2000）
- `agent/memory.py`：三层架构。L1 Working：`fact_index` dict lookup；L2 Episodic：`semantic_match()` 128d 余弦相似度（Hard>0.85 跳检索，Soft 0.5–0.85 作 retrieval_priors）；L3 Semantic：`global_facts` 表，≥3 hits + verified 晋升
- `_encode_text_lite`：复用 ColQwen text encoder 128d float16，fallback char 2-gram hash
- `conv_facts` 升级：`+fact_embedding BLOB, +grounding_verified, +hit_count, +tainted`
- grounding 反馈闭环：校验通过 → verified=1；未匹配 → tainted=1；满 3 次 verified → 晋升 global_facts
- `_check_known_facts` 从关键词 `in` 改为语义余弦匹配，"贵州茅台酒股份有限公司" 能匹配到 "茅台"

**Phase 3 — 执行层（P29/P30）**：
- Executor DAG 化：`SubTask +task_id/depends_on/priority/status`，构建依赖图 → 同层无依赖 `ThreadPoolExecutor(max_workers=5)` 并发。单 task 失败不中断整图，下游自动标记 failed。结果 `_merge_results()` 合并
- `plan_critic`：信号词触发（口径变更/任务失败），~200 token LLM 判断修订 → `{revise, new_subtasks, drop_ids}`
- `agent/prompts/` 重构为子目录：`planner/{base,single_fact,cross_doc_compare,multi_step_calc,trend_analysis}` + `few_shot/examples.jsonl`；`verifier/{base,numeric,strict,factual}`；`synthesizer/{base,numeric}`
- `load_prompt(node, variant, with_few_shot)`：按 variant 选文件，不存在 fallback base → legacy flat
- Planner 两段式：Stage1 `_classify_query`（heuristic 关键词 → ~200 token LLM）→ Stage2 `load_prompt(variant + few_shot)` → 生成 plan

**Phase 4 — 质量层（P31）**：
- Parallel Verification：`query_class ∈ {multi_step_calc, cross_doc_compare}` 且首轮 reflexion 时，3 实例（strict/base/numeric）`ThreadPoolExecutor` 并发
- 多数表决 `is_sufficient = votes > total/2`，`confidence = votes/total`；`missing_facts` 取并集按 `(sub_task_idx, root_cause)` 去重

**Phase 5 — 抽象层（P32）**：
- `skills/*.yaml`：4 个技能定义（single_fact/cross_doc_compare/multi_step_calc/trend_analysis），含 triggers/regex、plan_template、tools、strategy
- `skills/registry.py`：`SkillSpec` dataclass + `match_skill(query)`（keyword trigger O(1) → semantic fallback 留口）
- Planner Stage0：优先 `match_skill`，命中后 `query_class = skill.plan_template` + 策略注入

### 4.4 贯穿设计原则

1. **渐进式增强**：每个 Phase 在已有基础上叠加，不推倒重来。P26 在现有 API 上加 retry，P27 在 `_render_history` 上加压缩，P28 在 `_check_known_facts` 上改匹配算法
2. **LLM 判，规则修**：Planner/Verifier 用 LLM 做语义判断，remediation/grounding/fact_extractor 用确定规则执行
3. **门控控制成本**：P31 并行验证仅计算/对比类启用；P29 plan_critic 仅信号词触发；P27 压缩仅在超阈值触发
4. **纯规则优先**：能不用 LLM 的地方坚决不用——fact_extractor（regex）、grounding（集合查）、compress_history（正则启发式）、budget（计数器）
5. **配置化优于代码化**：prompt variant + skill definition 都是文件级配置，改 prompt 不需重新部署
6. **向后兼容**：新字段 Optional、新路径 fallback 旧路径、旧 flat prompt 文件保留作 fallback

---

## 五、关键设计亮点

- **视觉检索绕过 OCR**：ColQwen 多向量 + MaxSim 保留表格/图表布局，不经过文本分块
- **结构化根因诊断**：4 种 root_cause 驱动差异化修复，不做盲目 reflexion
- **三层记忆架构**：Working(单query) → Episodic(跨turn 语义匹配) → Semantic(跨conv 验证晋升)，零新依赖
- **DAG 并发执行**：依赖图 + 同层并行，两个独立检索 2T→T
- **动态 prompt 路由**：query_class 驱动 4 种 planner variant + few-shot，替代单一静态模板
- **多实例表决**：数值题 3 verifier 并行多数表决，confidence 连续值
- **全链路容错**：tenacity 指数退避 + transient/fatal 分类 + error_log 可观测
- **金融领域适配**：AST 安全求值、口径消歧、引用溯源 + grounding 后置审计
- **三层服务边界清晰**：ColQwen Service / Qdrant / FastAPI / Chainlit，任一层可独立替换

---

## 六、文件改动总览（P26–P32）

| 文件 | Phase | 内容 |
|------|-------|------|
| `agent/retry.py` | P26 | **[新]** with_retry + classify_error |
| `agent/compression.py` | P27 | **[新]** compress_history + compress_evidence + TokenBudget |
| `agent/memory.py` | P28 | **[新]** semantic_match + _encode_text_lite + promotion |
| `agent/nodes/plan_critic.py` | P29 | **[新]** 信号词触发 plan 修订 |
| `skills/registry.py` | P32 | **[新]** SkillSpec + match_skill |
| `skills/*.yaml` | P32 | **[新]** 4 个技能定义 |
| `agent/prompts/` | P30 | 重构为子目录 + 变体 + few_shot |
| `agent/state.py` | P26/29/30/31 | +ErrorLogEntry +TodoItem +error_log +todo_items +todo_updates +query_class +agent_profile；SubTask DAG 字段 |
| `agent/schemas.py` | P26.5/P29 | +TodoItemSchema；PlanItemSchema DAG 字段 |
| `agent/llm.py` | P26 | timeout=60, max_retries=3 |
| `agent/graph.py` | P29 | +plan_critic 节点 + 条件边 + 初始 state 扩展 |
| `agent/nodes/executor.py` | P26/28/29 | error_log + todo 追踪 + 语义 cache + DAG 调度 |
| `agent/nodes/planner.py` | P27/30/32 | compress_history + 两段式 + skill 匹配 |
| `agent/nodes/verifier.py` | P27/30/31 | compress_evidence + variant + 三投表决 |
| `agent/nodes/synthesizer.py` | P30 | variant 选择 |
| `agent/nodes/grounding.py` | P28 | 反馈写回 conv_facts |
| `agent/nodes/remediation.py` | P26.5 | retry TodoItem + attempt++ |
| `agent/prompts/__init__.py` | P30 | variant + few_shot loader |
| `tools/colpali_tool.py` | P26 | httpx + @with_retry |
| `tools/vlm_tool.py` | P26 | @with_retry（仅 transient） |
| `backend/server.py` | P26/26.5 | SSE event:todo + error retryable |
| `backend/storage.py` | P26.5/28 | turn_todos + conv_facts 升级 + global_facts + 迁移 |
