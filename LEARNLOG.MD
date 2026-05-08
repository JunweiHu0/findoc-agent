# FinDoc Agent 实现详解

## 一、项目概述

FinDoc Agent 是一个面向**中文金融文档（年报、招股书、研报）**的多模态 RAG Agent。

**核心思路**：绕过传统 OCR + 文本分块 pipeline，直接用**视觉检索（ColQwen2 多向量 + MaxSim）**定位相关页面图片，再用 **VLM（Qwen-VL）**从图片中读取内容，最后由 **LLM（DeepSeek）**规划、验证、合成带引用的答案。

**与传统 RAG 的本质区别**：

```
传统 RAG:  PDF → OCR → 文本分块 → 单向量 embedding → 语义搜索 → LLM 回答
FinDoc:    PDF → 页面图片 → ColQwen2 多向量索引 → MaxSim 视觉检索 → VLM 读图 → LLM 回答
```

金融文档表格密集、版式复杂。OCR 在表格区域出错率高，文本分块会割裂表格行列关系。ColQwen2 直接用视觉编码器处理页面图片，完整保留了表格、图表、数字的空间布局信息。

---

## 二、基础概念

### 2.1 MaxSim — 多向量后期交互检索

传统文本检索用单向量（一个文档 → 一个 768/1024 维向量），查询和文档各一个向量，算余弦相似度即可。

ColQwen2 属于**多向量（multi-vector）**模型，也称为**后期交互（late interaction）**或 **ColBERT-style** 检索：

- **编码阶段**：每张页面图片经过 ColQwen2 视觉 encoder，输出一个 `[N_patches, D]` 的矩阵。N_patches ≈ 1024（每页图片被切为 1024 个 patch），D = 128。每页不是 1 个向量，而是 **1024 个向量**。
- **查询阶段**：文本查询经过 ColQwen2 文本 encoder，输出 `[T_tokens, D]` 矩阵。T_tokens ≈ 查询的 token 数（通常 10–50），D = 128。
- **MaxSim 打分**：对查询的每个 token，在页面的所有 patch 中找最相似的那个，取最大值；然后把所有 token 的最大值求和：

```
score(query, page) = Σ_{i=1..T} max_{j=1..N} ⟨q_i, d_j⟩
```

其中 q_i 是查询第 i 个 token 的向量，d_j 是页面第 j 个 patch 的向量，⟨·,·⟩ 是内积。

**Python 实现**（`tools/colpali_tool.py:_maxsim`）：

```python
def _maxsim(query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
    # query_emb: [T, D]  doc_emb: [P, N, D]
    q = query_emb.float()
    d = doc_emb.float()
    sim = torch.einsum("td,pnd->ptn", q, d)   # [P, T, N] pairwise similarities
    return sim.max(dim=-1).values.sum(dim=-1)  # [P] final scores
```

**为什么叫"后期交互"**：查询和文档的向量在检索时才第一次交互（计算内积），而"早期交互"模型（如 cross-encoder）会把 query 和 doc 拼接后一起过 transformer。"后期"意味着文档向量可以**离线预计算**、存入向量库，检索时只编码查询。

**存储量级**：一份 200 页年报 ≈ 200 × 1024 × 128 × 2 bytes（float16）= 约 50MB。单向量库（Chroma/pgvector）不支持多向量存储，必须用 Qdrant 1.10+（`MultiVectorConfig`）或 Milvus 2.5+。

**Qdrant 服务端 MaxSim**：Qdrant 的 `MultiVectorConfig(comparator=MAX_SIM)` 将 MaxSim 计算下推到数据库内部执行，不需要把全部向量拉回客户端。使用 `Distance.DOT`（内积）而非 `Distance.COSINE`——COSINE 会先对向量做 L2 归一化，导致 MaxSim 结果与 Python 实现不一致。

### 2.2 SSE — Server-Sent Events

SSE 是 HTTP 长连接协议，服务器向客户端**单向**推送事件流。相比 WebSocket，SSE 更简单（纯 HTTP，不需要升级协议），适合"服务器持续推送进度"的场景。

**协议格式**：
```
event: node
data: {"node":"planner","summary":"3步 [跨文档对比]: ...","content":"..."}

event: status
data: {"type":"status","message":"VLM 正在读取 p005.png..."}

event: done
data: {"type":"done","answer":"...","grounding_score":0.95}

: keepalive    ← 空事件，维持连接
```

**FinDoc 的 SSE 架构**：

```
┌─ 后台线程 ─────────────────────────────────────┐
│  LangGraph app.stream(state)                     │
│    → 每个节点完成后 put(node_queue, (name, delta))│
│    → 工具内部 _report_progress(msg) put(progress) │
│    → synthesizer 流式 _push_token(t) put(token)  │
└──────────────────┬──────────────────────────────┘
                   │ queue.Queue (线程安全)
                   ▼
┌─ 主 async 循环 ─────────────────────────────────┐
│  while True:                                     │
│    drain progress_queue → yield event: status     │
│    drain token_queue    → yield event: token      │
│    drain node_queue     → yield event: node       │
│    if __done__: break                            │
│    await asyncio.sleep(0.3)                      │
└─────────────────────────────────────────────────┘
                   │
                   ▼
            Chainlit 前端 (httpx.AsyncClient.stream)
```

关键设计：LangGraph 在后台线程同步执行（因为 ColQwen/VLM 调用是同步阻塞的），主 async 循环通过 `queue.Queue`（线程安全）轮询结果。这种"同步线程生产 + 异步循环消费"模式让长耗时的模型调用不会阻塞 HTTP 响应。

### 2.3 Reflexion — 自我修正循环

Reflexion 是 Shinn et al. (2023) 提出的 Agent 自我修正范式。在 FinDoc 中的实现：

```
executor → verifier → 不充分? → remediation → executor → verifier → ...
                         │
                         充分? → synthesizer
```

Verifier 判断证据是否充分。不充分时，不是简单地"再来一次"，而是**诊断根因**：

| root_cause | 含义 | 修复策略 |
|---|---|---|
| `retrieval_miss` | ColQwen 没召回正确页面 | 放宽 top_k / 改写 query 重新检索 |
| `reading_miss` | 页面召回了但 VLM 读错/漏读 | 精炼 VLM 指令重读同一批页面，不跑检索 |
| `ambiguous_query` | 原 sub_query 歧义 | 改写为 self-contained query |
| `inconsistency` | 跨页数字不一致 | 触发口径消歧（合并报表 vs 母公司） |

三重保护防止死循环：`max_reflexion_iter=3` + `budget_retrievals=10` + `budget_vlm_calls=20`。超出任一上限强制进入 synthesizer，标记低置信。

### 2.4 ColQwen2 — 视觉检索模型

ColQwen2 是基于 Qwen2-VL 微调的 ColBERT-style 多向量检索模型。支持两种模态输入：

- **图像编码**：页面图片 → vision encoder → `[N_patches, 128]` 多向量
- **文本编码**：查询文本 → text encoder → `[T_tokens, 128]` 多向量

两个 encoder 共享同一个向量空间（128 维），可以直接做内积。这与 CLIP-style 模型（单向量）的本质区别是**保留了细粒度的 patch 级信息**——对于含有多个表格、不同区域数字的财报页面，MaxSim 能精确匹配到"包含答案的那几个 patch"。

项目默认 backbone 是 `vidore/colqwen2-v0.1`，用户有基于 ColQwen2 微调的 LoRA 可叠加。

---

## 三、系统架构

### 3.1 8 节点 Agent 图

```
retrieval_scout → planner → executor ──→ plan_critic? ──→ executor (revised)
                    ▲          ▲      │                    │
                    │          │      ▼                    ▼
                    │          │    verifier ──┐          verifier
                    │          │      │ suff?   │
                    │          └── remediation ←┘ not
                    │                            │
                    └── plan revision ───────────┘ (signal triggered)
                                                 │
                                          synthesizer → grounding → END
```

### 3.2 节点职责速查

| 节点 | 文件 | 核心逻辑 |
|------|------|---------|
| `retrieval_scout` | `agent/nodes/planner.py` | 预检索全库 top-3 文档，planner 据此做 target_doc 决策 |
| `planner` | `agent/nodes/planner.py` | 两段式：classify query → variant prompt + few-shot → 生成 DAG plan |
| `executor` | `agent/nodes/executor.py` | DAG 拓扑调度，同层并发执行（ThreadPool max_workers=5） |
| `plan_critic` | `agent/nodes/plan_critic.py` | 信号词/任务失败触发，LLM 评估剩余 plan 合理性 |
| `verifier` | `agent/nodes/verifier.py` | 充分性判断 + 数值题并行 3 实例多数表决 |
| `remediation` | `agent/nodes/remediation.py` | 按 root_cause 分派修复策略，budget 控制 |
| `synthesizer` | `agent/nodes/synthesizer.py` | 证据 → 带引用答案，流式 token-by-token |
| `grounding` | `agent/nodes/grounding.py` | citation 反向校验 + numeric fuzzy match + 反馈写回 |

### 3.3 模型分工

| 模型 | API | 节点 | 作用 |
|------|-----|------|------|
| DeepSeek V4 Flash | OpenAI-compat, 文本 | planner/verifier/synthesizer/plan_critic | 规划、验证、合成、修订 |
| ColQwen2 | 本地 GPU / Litserve 服务 | retrieval_scout/executor 检索阶段 | 视觉多向量检索 |
| Qwen VLM | DashScope API, 多模态 | executor VLM 读页阶段 | 页面图片 → 结构化文本 |

### 3.4 部署拓扑

```
┌─ FastAPI (:8001) ───────────────────────────────┐
│  POST /api/v1/query        SSE 流式输出          │
│  GET  /api/v1/health                             │
│  GET  /api/v1/conversations  CRUD 历史对话        │
│  POST /api/v1/upload         上传文档建索引        │
│  startup → 预加载 ColQwen2 + 索引 (warm-up)       │
└──────────────┬───────────────────────────────────┘
               │
    ┌──────────┴────────────┐
    ▼                       ▼
┌─────────────┐    ┌──────────────────┐
│ ColQwen      │    │ Qdrant (:6333)    │
│ Litserve     │    │ MultiVectorConfig │
│ GPU :8000    │    │ MAX_SIM + DOT     │
└─────────────┘    └──────────────────┘

┌─ Chainlit frontend ────────────────────────────┐
│  httpx.AsyncClient.stream(POST /api/v1/query)   │
│  消费 SSE: event=node|status|token|done|error    │
│  UI: Steps + inline Image + 置信度 banner        │
└────────────────────────────────────────────────┘
```

四种 config 组合：

| `services.colqwen_url` | `retriever.backend` | 说明 |
|---|---|---|
| `""` | `in_memory` | 默认：本地模型 + 本地 .pt MaxSim |
| `url` | `in_memory` | 远程编码 + 本地 MaxSim |
| `""` | `qdrant` | 本地编码 + Qdrant 服务端 MaxSim |
| `url` | `qdrant` | 完全分布式（生产推荐） |

---

## 四、数据模型

### 4.1 AgentState — 共享状态

所有节点通过 `AgentState`（TypedDict）共享数据。带 `Annotated[T, add]` 标注的字段使用 LangGraph 的 **reducer 模式**——节点返回 delta，框架自动 append 到已有列表，无需手动合并。

```python
class AgentState(TypedDict, total=False):
    query: str                                    # 用户原始问题
    plan: list[SubTask]                           # planner 生成的执行计划
    plan_cursor: int                              # executor 当前执行位置

    # 累积证据（Annotated[..., add] — reducer 自动合并）
    retrieved_pages: Annotated[list[PageHit], add]
    extracted_facts: Annotated[list[Fact], add]
    computed_values: Annotated[list[ComputedValue], add]

    # Reflexion 控制
    reflexion_iter: int                           # 当前反射轮次
    is_sufficient: bool                           # verifier 判决
    missing_facts: Annotated[list[dict], add]      # 结构化缺失信息
    tried_queries: Annotated[list[str], add]       # 已尝试的查询
    budget_retrievals: int                         # 检索预算剩余
    budget_vlm_calls: int                          # VLM 调用预算剩余

    # 记忆系统
    fact_index: dict                               # L1 Working: (entity,period,metric) → Fact
    known_facts: list[dict]                        # L2 Episodic: 跨轮缓存
    chat_history: list[dict]                       # 多轮对话历史
    doc_filter: Optional[list[str]]                # 用户上传文档过滤

    # 工程化（P26+）
    error_log: Annotated[list[dict], add]          # 结构化错误记录
    todo_items: Annotated[list[dict], add]         # 运行时任务状态
    query_class: str                               # P30: 查询分类
    agent_profile: dict                            # P31: specialist 路由预留

    # 最终输出
    answer: str
    citations: list[Citation]
    grounding_score: float
    unverified_claims: Annotated[list[dict], add]
```

### 4.2 SubTask — 执行计划单元

```python
class SubTask(BaseModel):
    sub_query: str                              # 自然语言检索词 或 算术表达式
    target_doc: Optional[str] = None            # 限定检索的 doc_id
    expected_output_schema: str = "text"        # "text"|"number"|"table"
    tool_calls: list[dict] = []                 # P21: 显式工具调用 [{tool, args}]

    # P29 DAG 字段
    task_id: str = ""                           # 唯一标识
    depends_on: list[str] = []                  # 依赖的 task_id 列表
    priority: Literal[0, 1] = 0                # 0=核心, 1=探索
    status: Literal["pending","running","done","failed"] = "pending"
```

### 4.3 Fact — 结构化事实

```python
class Fact(BaseModel):
    text: str                                   # 原始文本
    source_doc: str                             # 来源文档
    source_page: int                            # 来源页码
    sub_task_idx: Optional[int] = None          # 关联的 plan 步骤

    # P24 结构化抽取字段（fact_extractor 纯 regex 填充）
    entity: Optional[str] = None                # "贵州茅台"
    period: Optional[str] = None                # "2023" / "2023H1"
    metric: Optional[str] = None                # "营业收入" / "毛利率"
    value: Optional[float] = None               # 1505.6
    unit: Optional[str] = None                  # "亿元"
    raw_kind: Literal["numeric","string","table_row","unstructured"] = "unstructured"
```

### 4.4 TodoItem — 运行时任务追踪 (P26.5)

```python
class TodoItem(BaseModel):
    id: str                                     # t-{uuid}
    sub_task_idx: int                           # 对应 plan[i]
    title: str                                  # 35 字截断标题
    status: Literal["pending","running","done","failed","skipped"]
    attempt: int = 0                            # 重试次数
    error: Optional[str] = None
    parent_id: Optional[str] = None             # retry 时指向原始 todo
```

---

## 五、节点实现详解

### 5.1 retrieval_scout — 检索前探查 (P22)

**问题**：planner 在没有任何检索信息的情况下就要决定 target_doc，只能凭 doc_id 字面（如 `moutai_2023`）猜测。用户上传的文档 doc_id 是 uuid 后缀（`balance_sheet_a3f9c1`），planner 几乎只能填 `target_doc=null`。

**方案**：在 planner 之前跑一次轻量预检索。对每个 doc 做 MaxSim → 取最高分的那一页 → 按最高分排序取 top-3 文档 → 把候选信息注入 planner prompt。

```python
def retrieval_scout_node(state: AgentState) -> dict:
    q_emb = _encode_query(query)
    # 对每个 doc 单独打分
    for doc_id, idx in indexes.items():
        scores = _maxsim(q_emb, idx["embeddings"])
        best_score = scores.max().item()
        # ...
    # 返回 top-3
    top_docs = sorted(doc_scores.values(), key=lambda d: d["top_score"], reverse=True)[:3]
    return {"scout_candidates": top_docs}
```

### 5.2 planner — 两段式规划 (P30)

**Stage 1 — 查询分类**：

先用关键词 heuristic 快速分（"对比/比较/vs" → `cross_doc_compare`，"趋势/历年" → `trend_analysis`），命中率 >95%。未命中用 ~200 token 轻量 LLM 调用分类。结果缓存到 `state.query_class`，后续轮次复用。

```python
def _classify_query(query: str, chat_context: str = "") -> str:
    if any(kw in query for kw in ["对比", "比较", "vs", "差异"]):
        return "cross_doc_compare"
    if any(kw in query for kw in ["趋势", "历年", "变化"]):
        return "trend_analysis"
    # ... heuristic fallback → lightweight LLM call
```

**Stage 2 — 变体 prompt + few-shot**：

根据 `query_class` 加载对应的 prompt template：

- `single_fact`：target_doc 优先，1–2 步即可
- `cross_doc_compare`：每公司独立 sub_task，depends_on 为空以便并行
- `multi_step_calc`：先 text 取数 → 再 number 算式，depends_on 标记依赖
- `trend_analysis`：每年一个 sub_task，放宽 top_k=8

如果 P32 skill registry 有匹配，优先使用 skill 的 plan_template 覆盖 query_class。

```python
# P32 Stage 0: match skill
skill = _match_skill(query)
if skill and skill.plan_template != "base":
    query_class = skill.plan_template

# Stage 2: load variant + few_shot
variant_prompt = load_prompt("planner", variant=query_class, with_few_shot=True)
result: PlannerOutput = llm.invoke(variant_prompt.format(...))
```

### 5.3 executor — DAG 并发执行 (P29)

**核心变化**（相比旧版 plan_cursor++ 串行）：

```python
def executor_node(state: AgentState) -> dict:
    plan = state.get("plan") or []

    # 1. 构建 DAG：determine completed / failed tasks
    completed = {t["id"] for t in existing_todos if t["status"] == "done"}
    failed    = {t["id"] for t in existing_todos if t["status"] == "failed"}

    # 2. 找所有就绪 task：所有 depends_on 已完成
    runnable = []
    for i, st in enumerate(plan):
        if task_ids[i] in completed or task_ids[i] in failed:
            continue
        if all(dep in completed for dep in st.depends_on):
            runnable.append((i, st))
        if any(dep in failed for dep in st.depends_on):
            failed.add(task_ids[i])  # 上游失败 → 下游自动失败

    # 3. 同层并发
    with ThreadPoolExecutor(max_workers=min(len(runnable), 5)) as pool:
        futures = {pool.submit(_execute_one_task, st, idx, state): (idx, st)
                   for idx, st in runnable}
        for fut in as_completed(futures):
            results.append(fut.result())

    # 4. 合并
    return _merge_results(results, plan)
```

**单 task 执行**（`_execute_one_task`）：
1. `tool_calls` 非空 → 按 Tool Registry dispatch
2. `expected_output_schema == "number"` → calculator AST 安全求值
3. 其他 → ColQwen 检索 + VLM 读页（并发 max 5 workers）+ fact_extractor 结构化

**并发控制**：
- 跨 task 并发：同层无依赖 task 通过 `ThreadPoolExecutor` 并发（外层）
- task 内并发：单 task 的多个页面 VLM 读取也通过 `ThreadPoolExecutor` 并发（内层）
- `max_workers=5`：两级并发共享 5 个 worker 上限

### 5.4 plan_critic — 按需计划修订 (P29)

**触发条件**（`_should_trigger`）：
1. `extracted_facts` 含信号词："口径变更/合并范围调整/重述会计政策/追溯调整"等
2. 任何 task `status == "failed"`

**行为**：调 ~200 token LLM 评估剩余 plan 是否合理，返回 `{revise: bool, new_subtasks: [...], drop_task_ids: [...]}`。revise=false 或 LLM 失败 → 保留原 plan。

```
executor → plan_critic? → executor  (修订后重新执行)
           plan_critic? → verifier  (无需修订继续)
```

### 5.5 verifier — 充分性验证 + 并行表决 (P19/P31)

**标准路径**（单实例）：LLM 判断 `is_sufficient` + `confidence` + 结构化 `missing_facts[{sub_task_idx, what, root_cause, suggested_*}]`。

**P31 并行验证路径**：`query_class ∈ {multi_step_calc, cross_doc_compare}` 且 `reflexion_iter == 1` 时启用。

```python
def _parallel_verify(state, plan, cursor, iter_count, query_class) -> dict:
    variants = ["strict", "base", "numeric"]  # 三个视角
    with ThreadPoolExecutor(max_workers=3) as pool:
        # 并发调用三个 verifier
        for variant in variants:
            prompt = load_prompt("verifier", variant=variant).format(...)
            results[variant] = llm.invoke(prompt)

    # 多数表决
    votes = sum(1 for r in results.values() if r.is_sufficient)
    is_sufficient = votes > len(results) / 2
    confidence = votes / len(results)

    # missing_facts 取并集，按 (sub_task_idx, root_cause) 去重
```

**早停**：`reflexion_iter > 1` 且本轮未产生新事实 → 强行 sufficiency，标记低置信 0.3。

### 5.6 remediation — 差异化修复 (P20/P26.5)

按 `root_cause` 分派四种修复策略：

```
retrieval_miss  → 放宽 top_k，改写 query 重检索
reading_miss    → 精炼 VLM instruction 重读同一批页（不跑 ColQwen）
ambiguous_query → 改写 sub_query 为 self-contained
inconsistency   → 触发 disambiguate_caliber 提取口径
```

每种修复消耗 budget。单次 inconsistency 消耗 2 个 VLM budget（需多页对比）。Budget 耗尽 → 强制 fallthrough。

**P26.5 retry 追踪**：每个修复产生新 `TodoItem`，`attempt++`，`parent_id` 指向原始 todo，形成完整执行链路树。

### 5.7 synthesizer — 流式答案生成 (P16/P30)

```python
def synthesizer_node(state: AgentState) -> dict:
    # P30: variant selection
    syn_variant = "numeric" if query_class in ("multi_step_calc", "cross_doc_compare") else "base"
    prompt = load_prompt("synthesizer", variant=syn_variant).format(...)

    if _token_hook is not None:
        # Streaming mode: yield tokens one by one for SSE
        for chunk in llm.stream(prompt):
            _push_token(chunk.content)
    else:
        msg = llm.invoke(prompt)
        return {"answer": msg.content, "citations": citations}
```

`_token_hook` 由后端 server 注册，将 token 写入 `token_queue`。前端逐字渲染实现"打字机效果"。

### 5.8 grounding — 后置事实校验 (P23/P28)

**纯规则节点，零 LLM 调用**：

1. **Citation 校验**：正则提取 `[doc_id p.X]`，逐一查 `extracted_facts` 中是否存在对应 (doc_id, page_num)。不存在 → 标记 `unverified`，从最终答案中剥离。

2. **Numeric 校验**：正则提取答案中的数字 + 单位，与 `extracted_facts` 中的数值做 ±0.1% fuzzy match。不匹配 → 标记 `unverified`。

3. **置信度 banner**：`grounding_score >= 0.95` 绿色通过；`0.7–0.94` 黄色警告；`<0.7` 红色严重。banner 列出具体未匹配项。

4. **P28 反馈写回**：校验通过的 fact → `grounding_verified=1`；未通过 → `tainted=1`。`hit_count ≥ 3 AND verified=1` → 自动晋升到 `global_facts`（L3 语义记忆）。

---

## 六、工具系统 (Tool Registry, P21)

### 6.1 设计

每个工具是一个 `ToolSpec`：

```python
@dataclass
class ToolSpec:
    name: str                   # "retrieve_pages"
    description: str            # planner prompt 中的工具说明
    params_schema: type[BaseModel]   # pydantic 输入校验
    output_schema: type[BaseModel]   # pydantic 输出校验
    callable: Callable          # 实际函数
    category: str               # retrieval|reading|compute|resolution
```

planner prompt 自动注入工具列表，executor 做纯 dispatch + schema 校验。

### 6.2 四种工具

| 工具 | category | 实现 |
|------|----------|------|
| `retrieve_pages` | retrieval | `colpali_retrieve(query, top_k, doc_filter)` → `[{doc_id, page_num, score, image_path}]` |
| `read_page_with_vlm` | reading | `vlm_read_page(image_path, instruction)` → `{extracted_text}` |
| `calculate` | compute | AST 安全解析算术表达式，白名单 `+ - * / ** // % ( )`，拒绝任意代码执行 |
| `disambiguate_caliber` | resolution | 重读冲突页面，提取口径元数据（合并/母公司/FY/CY），启发式挑选权威值 |

### 6.3 扩展方式

```python
# 添加新工具只需:
register(ToolSpec(
    name="my_tool",
    description="...",
    params_schema=MyInput,
    output_schema=MyOutput,
    callable=my_function,
    category="retrieval",
))
```

planner 在下一次 prompt 渲染时自动感知新工具。

---

## 七、记忆系统 (P28)

三层架构，从即时到持久，从窄到广：

```
L1 Working Memory (单 query 内)
  └─ state.fact_index: dict[(entity, period, metric) → Fact]
  └─ Executor/Verifier 用 dict lookup 判断"是否已有 X"
  └─ 生命周期：单次 query

L2 Episodic Memory (同 conversation 跨 turn)
  └─ conv_facts 表 + semantic_match() 余弦相似度
  └─ Hard hit (>0.85): 跳过检索，直接返回缓存 fact
  └─ Soft hit (0.5–0.85): 作为 retrieval_priors 注入 planner
  └─ 生命周期：同一 conversation

L3 Semantic Memory (跨 conversation)
  └─ global_facts 表
  └─ 进入条件: grounding_verified=1 AND hit_count >= 3
  └─ 生命周期：永久（直到手动清理）
```

### 7.1 语义匹配实现

```python
def semantic_match(sub_query, known_facts) -> dict:
    q_emb = _encode_text_lite(sub_query)          # 128d 向量
    for kf in known_facts:
        fact_text = f"{kf.entity} {kf.period} {kf.metric} {kf.text}"
        f_emb = _encode_text_lite(fact_text)       # 128d 向量
        score = cosine(q_emb, f_emb)
        if score >= 0.85:  hard_hits.append(kf)
        elif score >= 0.5: soft_hits.append(kf)
    return {"hard_hits": hard_hits, "soft_hits": soft_hits}
```

### 7.2 编码方案

- **首选**：ColQwen text encoder 128d float16（与检索 backbone 同一向量空间，语义一致性好）
- **Fallback**：char 2-gram hash → 128d 稀疏向量（冷启动 / 无 GPU 时，精度较低但不阻塞系统）

### 7.3 反馈闭环

Grounding 是唯一知道"答案是否真贴合证据"的节点。校验完成后回写：

```
citation 校验通过 → conv_facts.grounding_verified = 1
numeric 未匹配   → conv_facts.tainted = 1
hit_count >= 3 AND verified = 1 → upsert_global_fact()
```

闭环保证了 L3 语义记忆的质量：只有经过多次验证的事实才能进入跨对话记忆。

---

## 八、容错与可靠性 (P26)

### 8.1 重试机制

```python
# agent/retry.py
_TRANSIENT = (httpx.TimeoutException, httpx.ConnectError,
              openai.APITimeoutError, openai.RateLimitError,
              openai.APIConnectionError, openai.InternalServerError)

_FATAL = (openai.AuthenticationError, openai.PermissionDeniedError,
          openai.BadRequestError)

def with_retry(max_attempts=3, base=1.0, cap=30.0):
    """指数退避: 1s → 2s → 4s，上限 30s，仅重试 transient"""
    return retry(stop=stop_after_attempt(max_attempts),
                 wait=wait_exponential(multiplier=base, max=cap),
                 retry=retry_if_exception_type(_TRANSIENT),
                 reraise=True)
```

**覆盖范围**：

| 调用点 | 重试策略 |
|--------|---------|
| VLM API (`_call_openai_compat`) | `@with_retry` 内嵌，仅 transient。坏图不反复消耗额度 |
| ColQwen 远程编码 (`_encode_query_remote`) | `@with_retry` httpx POST，失败 fallback 本地模型 |
| LLM 节点调用 (`ChatOpenAI`) | `max_retries=3, timeout=60` langchain 内置 |

### 8.2 Error Log

所有节点在异常时调用 `classify_error(exc)` 生成结构化错误 `{error_type, message, retryable, fatal}`，追加到 `state.error_log`。前端收到 `event: error` 时根据 `retryable` 显示"可重试"/"不可重试"提示。

### 8.3 Budget 三重保护

```
max_reflexion_iter = 3    ← 反射轮次上限
budget_retrievals   = 10  ← 检索调用上限
budget_vlm_calls    = 20  ← VLM 调用上限
```

任一超限 → 强制 synthesizer，标记低置信。Budget 在 remediation 中递减，不同类型消耗不同：
- `retrieval_miss`：retrieval -1 + VLM -1
- `reading_miss`：仅 VLM -1（不跑检索）
- `inconsistency`：VLM -2（需多页对比）

---

## 九、前后端通信协议

### 9.1 SSE 事件类型

| event | 触发时机 | data 结构 |
|-------|---------|----------|
| `node` | 每个 LangGraph 节点完成 | `{node, summary, content}` |
| `status` | 工具长操作进行中 | `{type:"status", message}` |
| `token` | synthesizer 流式输出每个 token | `{type:"token", token}` |
| `todo` | 任务状态变更 (P26.5) | `{id, status, title, attempt}` |
| `error` | Agent 执行异常 | `{type:"error", message, retryable}` |
| `done` | 全流程完成 | `{type:"done", answer, citations, grounding_score, ...}` |

### 9.2 请求/响应流

```
POST /api/v1/query  {"query": "茅台2023年营收?", "conv_id": "abc123"}
    │
    ▼
后台线程:   app.stream(initial_state)
    │
    ▼ (queue.Queue)
主 async 循环:
    event: status  → "正在加载 ColQwen2 模型..."
    event: node    → {"node":"retrieval_scout","summary":"Top-3: moutai_2023","content":"..."}
    event: node    → {"node":"planner","summary":"1步 [单事实]: 茅台2023营收","content":"..."}
    event: todo    → {"id":"t-0","status":"running","title":"茅台2023年营业收入"}
    event: status  → "VLM 正在读取 p005.png..."
    event: node    → {"node":"executor","summary":"检索5页 抽取5事实 计算0","content":"..."}
    event: todo    → {"id":"t-0","status":"done"}
    event: node    → {"node":"verifier","summary":"充分 (置信100%)","content":"..."}
    event: token   → "贵州"
    event: token   → "茅台"
    ...
    event: node    → {"node":"grounding","summary":"校验通过 (100%)","content":"..."}
    event: done    → {"answer":"...","grounding_score":1.0,"citations":[...]}
```

---

## 十、上下文压缩 (P27)

### 10.1 历史压缩

`compress_history` 替代旧的 200 字硬截断。触发条件：>3 turns 或 >800 tokens。

```python
def compress_history(history, budget=600) -> str:
    if len(history) <= 6 and est_tokens <= 800:
        return _render_raw(history)   # 不触发，保持原文

    older = history[:-4]   # 旧轮次 → 结构化摘要
    recent = history[-4:]  # 最近 2 轮 → 保留原文

    # 正则提取 {entities, metrics, decisions, focus_doc}
    summary = _extract_summary(older)
    return f"[摘要] {summary}\n---\n{_render_raw(recent)}"
```

摘要示例：`[摘要] 实体: 贵州茅台, 宁德时代; 指标: 毛利率, 营收; 意图: 对比`

### 10.2 证据压缩

`compress_evidence` 在 verifier 渲染证据前调用。触发条件：>10 条 fact 或 reflexion_iter > 0。

```python
def compress_evidence(facts, query, budget=800) -> list:
    # 按 (entity, period, metric) 去重
    # 保留 text 最长的（信息量最大）
    # 与 query 无关的丢弃（但写入 episodic memory 可召回）
```

---

## 十一、动态 Prompt 系统 (P30/P32)

### 11.1 Prompt 目录结构

```
agent/prompts/
├── system/persona.txt                   # "你是金融文档 QA Agent..."
├── planner/
│   ├── base.txt                         # 通用（保留旧版行为）
│   ├── single_fact.txt                  # 单事实：target_doc 优先
│   ├── cross_doc_compare.txt            # 跨文档：并行 + 口径统一
│   ├── multi_step_calc.txt              # 多步计算：先取数再算
│   ├── trend_analysis.txt               # 趋势：放宽 top_k
│   └── few_shot/examples.jsonl          # 4 条示例
├── verifier/
│   ├── base.txt                         # 通用
│   ├── numeric.txt                      # 数值容差 ±0.1%
│   ├── strict.txt                       # 严格标准（P31 投票用）
│   └── factual.txt                      # 只看充分性
└── synthesizer/
    ├── base.txt
    └── numeric.txt                      # 公式优先 + 表格展示
```

### 11.2 加载逻辑

```python
def load_prompt(node: str, variant: str = "base", with_few_shot: bool = False) -> str:
    # 1. 找 {node}/{variant}.txt
    # 2. 不存在 → fallback {node}/base.txt
    # 3. 不存在 → fallback 旧扁平 {node}.txt
    # 4. with_few_shot → 从 examples.jsonl 筛选匹配 variant 的条目，前置注入
```

### 11.3 技能系统 (P32)

技能是 prompt variant + tool list + strategy 的**命名打包**。YAML 配置，非代码路径：

```yaml
# skills/cross_doc_compare.yaml
skill: cross_doc_compare
triggers: ["对比", "比较", "vs", "versus", "差异"]
plan_template: cross_doc_compare
tools: [retrieve_pages, read_page_with_vlm, calculate, disambiguate_caliber]
strategy:
  retrieve_top_k: 5
  parallel_fan_out: true
  caliber_check: required
```

Planner Stage 0 优先 `match_skill(query)`——keyword trigger O(1) 匹配。命中后 `query_class = skill.plan_template`，strategy 参数注入 `remediation_hint` 供 executor 消费。

---

## 十二、目录结构

```
findoc-agent/
├── agent/                          # Agent 核心（8 节点 + 基础设施）
│   ├── graph.py                    # LangGraph 图装配
│   ├── state.py                    # AgentState + SubTask/Fact/PageHit/TodoItem
│   ├── schemas.py                  # LLM 结构化输出 schema
│   ├── llm.py                      # DeepSeek ChatOpenAI 工厂
│   ├── config.py                   # config.yaml 加载 + 路径常量
│   ├── retry.py                    # P26: tenacity 重试装饰器
│   ├── compression.py              # P27: 上下文压缩 + TokenBudget
│   ├── memory.py                   # P28: 三层记忆 + 语义匹配
│   ├── nodes/
│   │   ├── planner.py              # retrieval_scout + planner（两段式）
│   │   ├── executor.py             # DAG 调度 + 检索/VLM/计算
│   │   ├── verifier.py             # 充分性验证 + 并行表决
│   │   ├── remediation.py          # 差异化修复 + budget 控制
│   │   ├── plan_critic.py          # P29: 按需计划修订
│   │   ├── synthesizer.py          # 流式答案生成
│   │   └── grounding.py            # citation + numeric 后置校验
│   └── prompts/                    # P30: 变体 prompt 目录
│       ├── __init__.py             # load_prompt(node, variant, few_shot)
│       ├── system/persona.txt
│       ├── planner/{base,single_fact,cross_doc_compare,...}
│       ├── verifier/{base,numeric,strict,factual}
│       └── synthesizer/{base,numeric}
├── tools/                          # 工具层
│   ├── registry.py                 # P21: ToolSpec + register + dispatch
│   ├── colpali_tool.py             # ColQwen2 检索（in_memory + Qdrant）
│   ├── vlm_tool.py                 # Qwen VLM 页面读取（带缓存）
│   ├── vlm_cache.py                # VLM 输出 SQLite 缓存
│   ├── calculator.py               # AST 安全求值
│   ├── disambiguate.py             # 口径消歧
│   └── fact_extractor.py           # P24: 正则结构化抽取
├── backend/                        # FastAPI 后端
│   ├── server.py                   # SSE 流 + REST endpoints
│   ├── storage.py                  # SQLite CRUD
│   └── schemas.py                  # API request/response models
├── app/                            # Chainlit 前端
│   ├── chainlit_app.py             # SSE 消费 + UI 渲染
│   └── data_layer.py               # 对话持久化（桥接到 backend/storage）
├── ingestion/                      # 文档摄取
│   ├── pdf_to_pages.py             # PDF → 页面图片
│   ├── build_index.py              # 离线建 ColQwen2 索引
│   ├── push_to_qdrant.py           # 索引导入 Qdrant
│   ├── upload.py                   # 用户上传 pipeline
│   └── model_loader.py             # ColQwen2 模型加载/编码
├── services/                       # 服务化
│   └── colqwen_server.py           # Litserve ColQwen 服务
├── skills/                         # P32: 技能定义
│   ├── registry.py                 # SkillSpec + match_skill
│   ├── single_fact.yaml
│   ├── cross_doc_compare.yaml
│   ├── multi_step_calc.yaml
│   └── trend_analysis.yaml
├── eval/                           # 评测
│   └── run_eval.py
├── config.yaml                     # 全局配置
├── requirements.txt
└── data/                           # 运行时数据
    ├── index/                      # ColQwen2 索引 (.pt)
    ├── pages/                      # 页面图片
    ├── reports/                    # 原始 PDF
    └── findoc.db                   # SQLite 数据库
```

---

## 十三、关键设计原则

1. **LLM 判，规则修**：Planner/Verifier 用 LLM 做语义判断；remediation/grounding/fact_extractor 用确定规则执行。LLM 负责"理解"，规则负责"行动"。

2. **纯规则优先**：fact_extractor（regex）、grounding（集合查 + fuzzy match）、compress_history（正则启发式）、budget（计数器）——能用规则就不调 LLM。

3. **门控控制成本**：P31 并行验证仅计算/对比类启用；P29 plan_critic 仅信号词触发；P27 压缩仅超阈值触发。

4. **渐进式增强**：每个 Phase 在已有基础上叠加，不推倒重来。P26 在现用 API 上加 retry，P27 在 `_render_history` 上加压缩，P28 在 `_check_known_facts` 上改匹配算法。

5. **向后兼容**：新字段 Optional、新路径 fallback 旧路径、旧 flat prompt 文件保留作 fallback。

6. **零新依赖**：tenacity（langchain 间接依赖）、tiktoken（唯一新增）、ColQwen 128d embedding（复用已有模型）、YAML（标准库 pyyaml）。

---

## 十四、代码阅读指南

按依赖关系从底层往上看，每一层只依赖它前面的层，没有循环依赖。

### 第一阶段：数据结构与配置

**1. `agent/state.py`** — 所有数据结构的定义

`AgentState`（TypedDict，贯穿全部节点共享）、`SubTask`（规划结果）、`Fact`（抽取出的结构化事实）、`PageHit`（检索命中的页面）、`MissingFact`（verifier 诊断缺失的 schema）、`TodoItem`（运行时任务追踪）。先看完这个，后面每个节点读写什么字段就都清楚了。

**2. `agent/config.py`** — 项目配置

模型名称、top_k、index 路径、预算上限（`MAX_REFLEXION_ITER = 3`、`TOP_K = 5`）。代码很短，读完知道全局常量从哪来。

**3. `agent/schemas.py`** — LLM 结构化输出的 pydantic 模型

`PlannerOutput`（计划结构体）、`VerifierOutput`（充分性判断 + `MissingFact` 列表）、`ToolCall`。这些是 LLM 按 `json_mode` 强制输出的格式。理解了这个，后面看 planner/verifier 的 prompt 就知道 LLM 在往什么结构填空。

### 第二阶段：基础设施

**4. `agent/llm.py`** — LLM 工厂

三个角色（planner/verifier/synthesizer）各自取 `ChatOpenAI` 实例，都走 DeepSeek API。没有抽象层，就是个按角色选模型的工厂函数，约 40 行。

**5. `tools/registry.py`** — 工具注册中心

定义了 `ToolSpec`（工具名 + 描述 + 入参出参 pydantic schema + 实际函数 + category）。四个内置工具：`retrieve_pages`（检索）、`read_page_with_vlm`（读图）、`calculate`（安全计算）、`disambiguate_caliber`（口径消歧）。planner 的 prompt 里注入 `get_tools_for_prompt()`，executor 用 `dispatch(name, args)` 调工具。看完这个再看 planner/executor 就知道 tool_calls 怎么被生成和执行。

### 第三阶段：工具层

**6. `tools/colpali_tool.py`** — 视觉检索核心

ColQwen2 多向量编码查询 → 对每份已索引文档做 MaxSim → 返回全局 top-k 页面。支持 in-memory 和 Qdrant 两种后端，Qdrant 失败自动 fallback 到 in-memory。启动期 `preload()` 消冷启动。

**7. `tools/vlm_tool.py`** — VLM 页面读取

页面图 + instruction → OpenAI 兼容接口（默认 Qwen-VL）→ 提取文本。有缓存层（`vlm_cache.py`，同图同指令免调）。API key 为空时返回 mock 结果，保证骨架可运行。

**8. `tools/fact_extractor.py`** — 结构化事实提取

纯正则引擎。从 VLM 返回的自由文本中抽取 entity（公司名，20+ 已知实体 + 通用模式）、period（年份/季度/半年）、metric（毛利率/营收/ROE 等 17 种指标）、value（数值）、unit（亿元/% 等单位）。输出决定后续 fact_index 查重和 grounding 数值校验的质量。

### 第四阶段：节点层（按执行顺序）

**9. `agent/nodes/planner.py`** — 包含两个节点

上半部分是 `retrieval_scout_node`：预检索全库，按 doc 聚合取每 doc 最佳页，返回 top-3 候选文档 + 相关性分数，让 planner 做 informed 决策而非盲规划。

下半部分是 `planner_node`：三阶段——① 匹配技能（YAML 驱动，关键词命中则覆盖策略）→ ② 分类查询（启发式关键词 >95% 命中，其余走 LLM）→ ③ 加载变体 prompt + few-shot → 生成 DAG 执行计划（含 `depends_on` 依赖和 `tool_calls`）。

**10. `agent/nodes/executor.py`** — DAG 并发调度

构建任务依赖图（`task_id → depends_on`），同层无依赖任务 `ThreadPoolExecutor` 并发执行（最多 5 并发）。上游失败 → 下游自动标记 failed。tool_calls 优先走 registry dispatch，否则走默认检索+VLM 路径。跨轮缓存检索前检查（`_check_known_facts`，语义匹配硬命中跳过检索）。VLM 读完后立即 `fact_extractor` 结构化抽取。

**11. `agent/nodes/plan_critic.py`** — 按需计划修订

两个触发条件 OR：① VLM 读到的文本含 8 个财务披露信号词（口径变更/合并范围调整/追溯调整等）② 有任务执行失败。触发后调 LLM 评估剩余计划是否合理，输出 `{revise, new_subtasks, drop_task_ids}`。正常执行路径下完全透明——是一个按需插入的旁路。

**12. `agent/nodes/verifier.py`** — 充分性判断

两条路径：① 标准单实例（常规查询）② 3 实例并行多数表决（数值/对比查询，strict/base/numeric 三个 variant 同时跑，少数服从多数）。输出结构化 `MissingFact` 列表，每个含 `root_cause`（4 类：检索遗漏/阅读遗漏/查询歧义/跨页不一致）和修复建议。内部有早停逻辑：多轮无新事实时强制合成。

**13. `agent/nodes/remediation.py`** — 差异化修复

按 root_cause 分派 4 条修复路径，每条扣减对应的检索或 VLM 预算：

| root_cause | 策略 | 扣减 |
|---|---|---|
| retrieval_miss | 放宽 top_k 重检索 | 检索 -1, VLM -1 |
| reading_miss | 精炼指令重读同批页面 | VLM -1 |
| ambiguous_query | 改写为自包含查询重检索 | 检索 -1, VLM -1 |
| inconsistency | 触发口径消歧 | VLM -2 |

预算耗尽时强制 `is_sufficient = True`，降级到 synthesis。

**14. `agent/nodes/synthesizer.py`** — 答案生成

汇总事实 → 带 `[doc_id p.N]` 引用生成最终答案。LLM 不可用时用 stub answer（原始证据拼接）。支持流式 token 输出：通过全局 `_token_hook` 回调注入，后端 server 将此 hook 指向 SSE 推送。

**15. `agent/nodes/grounding.py`** — 后置审计

纯规则，零 LLM 调用：
- 逐条检查答案中 `[doc_id p.N]` 引用是否真的在 `extracted_facts` 里出现过
- 逐条检查答案中的数值是否与证据中的数值模糊匹配（±0.1% 容差）
- 未通过的直接从答案中剥离，插入置信度 banner
- 校验结果回写数据库，用于记忆晋升判断

### 第五阶段：组装与辅助

**16. `agent/graph.py`** — 8 节点拓扑 + 三条条件边

约 80 行。`build_graph()` 添加 8 个节点，`compile_graph()` 编译为可 stream/invoke 的 runnable。三条条件边的路由函数：
- `_route_after_executor`：进 plan_critic 还是 verifier
- `_route_after_verifier`：进 synthesizer 还是 remediation
- `remediation → executor`（无条件的回流边）

**17. `agent/memory.py`** — 三层语义记忆

| 层 | 名称 | 机制 | 作用 |
|---|---|---|---|
| L1 | 工作记忆 | 同轮 `fact_index` dict 查找 | 同轮去重 |
| L2 | 情景记忆 | conv_facts + 128d embedding + cosine | 硬命中(>0.85)跳过检索，软命中(0.5-0.85)作为先验 |
| L3 | 语义记忆 | global_facts 跨对话 | promotion 条件：hit_count ≥ 3 且 grounding_verified = 1 |

embedding 复用 ColQwen 文本编码器，零额外依赖。模型未加载时回退到字符 2-gram hash 稀疏向量。

**18. `agent/compression.py`** — 上下文压缩

- `compress_history`：聊天历史 >3 轮或 >800 token 时，旧轮次摘要为 `{实体, 指标, 意图, 文档}`，仅保留最近 2 轮 verbatim
- `compress_evidence`：事实 >10 条时，按 `(entity, period, metric)` 去重，保留文本最长的一条
- `TokenBudget`：每个节点有独立的 token 上限（planner 1200 / verifier 1500 / synthesizer 2000）

**19. `backend/server.py`** — FastAPI 后端

- `POST /api/v1/query`：SSE 流式推送节点进度 + token 实时输出 + todo 状态更新
- 后台线程跑 LangGraph `stream()`，主线程 queue 轮询推 SSE — 解决 `astream` 在主线程的阻塞问题
- 启动预加载模型 + 索引消冷启动
- 对话持久化（SQLite）+ 跨轮事实复用 + 自动标题生成
- 文档管理 API：列表/删除/重索引/封面缩略图
- 文件上传 + 异步处理管道 + SSE 进度追踪

### 阅读路线总结

```
state → config → schemas → llm
  → tools/ (registry → colpali → vlm → fact_extractor)
  → nodes/ (planner → executor → plan_critic → verifier → remediation → synthesizer → grounding)
  → graph → memory → compression → backend/server
```
