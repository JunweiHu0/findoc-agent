# FinDoc Agent

> 面向视觉密集型金融文档（年报、招股书、研报）的多模态 RAG Agent。基于 **ColQwen2** 视觉检索 + **LangGraph** 8 节点状态机，具备结构化根因诊断、LLMCompiler 风格 DAG 执行、三层语义记忆和跨轮事实复用能力。

[![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-1c3c3c)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Chainlit](https://img.shields.io/badge/frontend-Chainlit-9c27b0)](https://docs.chainlit.io/)
[![ColQwen2](https://img.shields.io/badge/retriever-ColQwen2-ff6f00)](https://huggingface.co/vidore/colqwen2-v0.1)
[![Qdrant](https://img.shields.io/badge/vector_db-Qdrant-DC244C?logo=qdrant)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](./LICENSE)

**语言:** **简体中文** | [English](./README.md)

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [关键设计决策](#3-关键设计决策)
4. [项目结构](#4-项目结构)
5. [快速开始](#5-快速开始)
6. [技术栈](#6-技术栈)
7. [开发路线图](#7-开发路线图)
8. [参考文献](#8-参考文献)

---

## 1. 项目概述

### 问题背景

金融年报有两个特点让传统文本 RAG 表现很差：

1. **版式复杂、信息密集。** 资产负债表、现金流表、组织架构图、合并报表附注——文本分块后语义被切碎，OCR 还原表格结构损耗大，跨单元格关系丢失。
2. **跨页跨文档推理需求强。** 典型用户查询——"对比 A、B 两家公司近三年毛利率趋势"——本质是需要多步检索 + 数值计算 + 对比生成的任务，单轮 RAG 无法覆盖。

### 核心思路

**用视觉检索绕过 OCR，用 Agent 状态机编排多步推理。**

- **ColQwen2** 直接以**页面图像**为检索单位，多向量 + MaxSim 后期交互，完整保留表格结构、图表布局、数字格式——这些是 OCR 会破坏的信息。
- **Agent 层**负责查询路由、任务分解为 DAG 计划、LLMCompiler 风格跨任务数据流、工具调度、根因诊断、差异化修复、引用/数值审计——把"召回的页面"转化为"带引用且经过校验的结构化回答"。

### 当前能力

| 能力 | 说明 |
|---|---|
| **查询路由** | 关键词启发式 + 轻量 LLM（~80 token）判断是否启用检索或直接作答 |
| **多文档问答** | 跨 14+ 已索引年报自动路由查询 |
| **跨公司对比** | 支持口径消歧的跨公司指标对比 |
| **数值计算** | AST 安全表达式求值 + `$task_id.value` 跨任务数据流（LLMCompiler 风格） |
| **结构化反射** | 四类根因诊断：检索遗漏 / 阅读遗漏 / 查询歧义 / 跨页不一致 |
| **引用校验** | 从答案文本正则解析 `[doc p.N]` 引用，对比 evidence 集合剥离虚构引用 |
| **跨轮记忆** | 三层架构：工作 dict → 情景 SQLite → 语义全局；cosine ≥ 0.85 硬命中跳过检索 |
| **流式输出** | Synthesizer token-by-token SSE 实时推送 |
| **用户上传文档** | PDF/图片上传 → 自动编码 → Qdrant 索引 → 立即可查询 |
| **知识库管理** | 面板 UI：列表 / 删除 / 重新索引 / 封面预览 |
| **错误恢复** | Tenacity 指数退避，区分瞬时/致命错误，结构化 `error_log` |

---

## 2. 系统架构

### 2.1 Agent 工作流（8 节点状态机）

```
用户问题
    │
    ▼
┌──────────────────┐  关键词启发式 + LLM 兜底（~80 token）
│  query_router    │  判断：是否需要文档检索？
└────────┬─────────┘
         │
   ┌─────┴─────┐
   │ 需要     │
   │ 检索？    │
   └─────┬─────┘
   不需要 │      需要
         ▼         ▼
┌─────────────┐  ┌──────────────────┐  全库 MaxSim → top-3 候选文档
│ synthesizer │  │ retrieval_scout  │  （为 planner 提供候选文档上下文）
│  (直接作答)  │  └────────┬─────────┘
└──────┬──────┘           │
       │                  ▼
       │         ┌──────────┐         query_class 分类 → 变体 prompt
       │         │ planner  │  ──►    + few-shot → 有序 DAG plan（含 task_id
       │         └────┬─────┘         和 `$tN.value` 跨任务占位符）
       │              │
       │              ▼
       │         ┌──────────┐         DAG 拓扑排序 → ThreadPool 分层并发；
       │         │ executor │  ──►    `$tN.value` 占位符解析 → calculator；
       │         └────┬─────┘         Tool Registry 调度；VLM 读取 + fact_extractor
       │              │
       │              ▼
       │    ┌────────────────────┐
       │    │ 触发 plan_critic?  │  信号词或 failed todo → plan_critic
       │    │                    │  按需修订 plan（重入保护：cursor + iter 上限）
       │    └──┬──────────┬──────┘
       │   否  │          │ 是
       │       ▼          ▼
       │  ┌──────────┐  ┌─────────────┐  修订 plan → 回到 executor
       │  │ verifier │  │ plan_critic │  （最多修订 2 次）
       │  └────┬─────┘  └──────┬──────┘
       │       │                │
       │       │                └──→ executor
       │       ├─ 充分 ─────────────────────┐
       │       │                             ▼
       │       ├─ 不充分 ──► ┌──────────┐  根因分派 → 显式 tool_calls
       │       │            │remediation│  (read_page_with_vlm / disambiguate_caliber)
       │       │            └─────┬─────┘  预算扣减；耗尽 → 强制 fallthrough
       │       │                  │
       │       │                  ▼
       │       └─────────────→ executor
       │                              │
       └──────────────────────────────┘
                                      ▼
                             ┌─────────────┐
                             │ synthesizer │  汇总事实 → 引用答案（SSE 流式输出）
                             └──────┬──────┘  从输出文本正则解析 `[doc p.N]`；
                                    │         对比 evidence 剥离虚构引用
                                    ▼
                                   END
```

**8 个节点：** 7 个常驻（query_router、retrieval_scout、planner、executor、verifier、remediation、synthesizer）+ 1 个按需触发（plan_critic）。

### 2.2 部署架构

```
┌─ uvicorn backend.server:app (port 8001) ──────┐
│  FastAPI + SSE                                  │
│  POST /api/v1/query     → SSE 流（8 节点进度） │
│  GET  /api/v1/documents → 已索引文档列表        │
│  GET  /api/v1/health                            │
│  startup → 预加载 ColQwen2 + 索引               │
│  per-query → load/save conv_facts               │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴────────────┐
        ▼                       ▼
┌────────────────┐    ┌──────────────────────┐
│ ColQwen Service│    │   Qdrant (多向量)     │
│ Litserve + GPU │    │   Docker · port 6333  │
│ port 8000      │    │   collection:         │
└────────────────┘    │   findoc_pages         │
                      └──────────────────────┘

┌─ chainlit run app/chainlit_app.py ────────────┐
│  纯 UI 层（仅 import chainlit / httpx）         │
│  消费 SSE：event=status|token|node|todo        │
│  动态 Step + 引用 inline Image + TodoList      │
└────────────────────────────────────────────────┘
```

**关键边界：** `agent/`、`tools/`、`ingestion/` 是业务核心，由后端直接 import；前端仅通过 HTTP/SSE 调用，不 import 任何 agent 代码。

### 2.3 节点职责

| 节点 | 职责 |
|---|---|
| `query_router` | 关键词启发式 + ~80 token LLM 判定是否检索；图条件边据此路由检索 vs 直答 |
| `retrieval_scout` | 预检索全库，返回 top-3 候选文档 + 相关性分数，让 planner 做知情决策 |
| `planner` | 两段式：query_class 分类 → 变体 prompt + few-shot → 有序 DAG plan（含 `task_id`、`depends_on`、`$tN.value` 占位符） |
| `executor` | DAG 拓扑调度 + ThreadPool 同层并发；`$tN.value` 占位符解析 → calculator；Tool Registry 调度；VLM 读取 |
| `plan_critic` | 信号词或 failed todo 触发的按需 plan 修订；重入保护 `plan_critic_last_cursor` + `plan_critic_iter`（上限 2） |
| `verifier` | 结构化 `MissingFact[]`（含 4 类根因）；数值/对比类查询启用 3 实例并行投票（strict/base/numeric） |
| `remediation` | 根因分派 → 显式 tool_calls（非字符串前缀 hack）；三重预算保护（iter=3 / retrieval=10 / vlm=20） |
| `synthesizer` | 汇总事实 → `[doc p.N]` 格式引用答案（SSE 流式）；正则解析引用 → 对比 evidence 集合剥离虚构项 |

### 2.4 四种检索部署组合

| `services.colqwen_url` | `retriever.backend` | 含义 |
|---|---|---|
| `""` | `in_memory` | 默认开发模式：本地模型 + 本地 .pt MaxSim |
| `http://localhost:8000` | `in_memory` | ColQwen 服务化 + 本地 MaxSim |
| `""` | `qdrant` | 本地编码 + Qdrant 服务端 MaxSim |
| `http://localhost:8000` | `qdrant` | 完全分布式（推荐生产环境） |

任何 Qdrant 异常自动 fallback 到 `_in_memory_retrieve`——Agent 不会因向量库故障而崩溃。

---

## 3. 关键设计决策

### 3.1 为什么选 LangGraph 而非 LangChain AgentExecutor？

LangChain 的 AgentExecutor 是黑箱循环。LangGraph 是显式状态机——每个节点的 I/O 可观测，reflexion 循环通过条件边控制，整个拓扑在 `graph.py` 里仅 ~30 行装配代码。

### 3.2 为什么不用 ReAct？

金融 QA 是结构化任务（实体 × 周期 × 指标）。ReAct 的串行 think-act-observe 循环在跨公司对比时丢失并发、烧 token、丢了可观测性。"分解-执行"DAG + 带类型标注的节点产出是更适合的范式。

### 3.3 为什么用 ColQwen2 做视觉检索？

传统 RAG：`PDF → OCR → 文本分块 → 单向量 embedding → 语义搜索`。金融文档表格密集，OCR 出错率高，单向量压缩丢失空间布局信息。

ColQwen2 将每页图片编码为**多向量**（每 patch 一个向量，每页约 1024 个 × 128 维）。检索用 **MaxSim**——query 的每个 token 找文档中最相似的 patch，再求和：

```
score(q, d) = Σ_{i ∈ q_tokens} max_{j ∈ d_tokens} ⟨q_i, d_j⟩
```

表格结构、图表布局、数字位置信息完整保留。ViDoRe benchmark：nDCG@5 ~89%（ColPali ~81%）。2B 基座在 RTX 3060 6GB bf16 推理不会 OOM。

### 3.4 为什么必须用 Qdrant？

标准向量库（Chroma、Pinecone、pgvector）只支持单向量存一条记录。一份 200 页年报产生约 20 万个向量——Qdrant 1.10+ 是少数原生支持 `MultiVectorConfig(comparator=MAX_SIM)` 的数据库。使用 `Distance.DOT`（不用 COSINE）保证与 Python einsum 点积结果完全一致。

### 3.5 查询路由

并非所有查询都需要检索。"你好" / "你能做什么" / 基于对话历史可回答的追问——应跳过整条检索+规划+执行流水线。`query_router` 用关键词启发式识别强信号（如"2023年营收"→ 检索；"你好"→ 直答），模糊情况走 ~80 token LLM 判断。避免在闲聊轮次浪费检索+VLM 预算。

### 3.6 LLMCompiler 风格 DAG 执行

Planner 产出 DAG plan——每个 subtask 有 `task_id` 和 `depends_on` 列表。跨任务数据通过 `$tN.value` 占位符流转：下游任务调用 calculator 前，先用前置任务的输出替换占位符。Executor 做拓扑排序 + 同层 ThreadPool 并发。Synthesizer prompt 有硬约束："有 `compute:` 行就必须用其 value，禁止重算"。

### 3.7 结构化 Reflexion

Verifier 不再输出模糊的"还需要更多信息"字符串，而是结构化的 `MissingFact` 列表，每条带 `root_cause` 枚举：

| root_cause | 修复策略 |
|---|---|
| `retrieval_miss` | 放宽 `top_k`，用改写后的 query 重检索 |
| `reading_miss` | 构造显式 `read_page_with_vlm` tool_calls 重读同批页 |
| `ambiguous_query` | 改写为完全自包含的查询 |
| `inconsistency` | 触发 `disambiguate_caliber`，传入冲突事实文本 |

三重防死循环保护：`max_reflexion_iter=3` + `budget_retrievals=10` + `budget_vlm_calls=20`。

### 3.8 Tool Registry

工具自描述 `(name, description, params_schema, output_schema)`，planner prompt 自动感知可用工具，executor 纯 dispatch + 输出校验。加新工具只需一行 `register(ToolSpec(...))`——无需改 planner、executor 或 schemas。

| 工具 | 类别 | 功能 |
|---|---|---|
| `retrieve_pages` | retrieval | ColQwen2 多向量 MaxSim 检索 |
| `read_page_with_vlm` | reading | VLM 页面图 → 结构化文本 |
| `calculate` | compute | AST 受限安全数值求值（支持 `$tN.value` 解析） |
| `disambiguate_caliber` | resolution | 跨页数值冲突 → VLM 口径提取 |

### 3.9 引用校验

Synthesizer 输出后，纯规则校验（正则 + 集合查，零 LLM 调用）：从答案文本用 `[doc p.N]` 模式匹配解析引用，再与 `extracted_facts` 证据集合对比。虚构引用直接被剥离。只有模型真正引用的页才返回给前端。

### 3.10 跨轮事实记忆（三层）

- **工作记忆**：`fact_index` dict `{(实体, 期间, 指标): Fact}` — 单次查询内有效
- **情景记忆**：SQLite `conv_facts` 表 + ColQwen 文本编码器 128d float16 向量 — 跨对话轮次复用；`known_facts` 在检索+VLM 前优先检查
- **语义记忆**：`hit_count ≥ 3` 且 `grounding_verified=1` 的事实晋升为 `global_facts`，跨对话复用

下轮追问时 executor 先查 `known_facts`——若 `(茅台, 2023, 营收)` 已命中，直接跳过 retrieval+VLM。连续追问场景预期减少 40–60% 检索调用。

---

## 4. 项目结构

```
findoc-agent/
├── agent/                       # Agent 核心 — 8 节点 LangGraph 状态机
│   ├── graph.py                 #   build_graph() — 30 行拓扑 + 条件边
│   ├── state.py                 #   AgentState TypedDict + Fact / SubTask / PageHit / Citation
│   ├── schemas.py               #   LLM 结构化输出 schema（PlannerOutput, VerifierOutput）
│   ├── config.py                #   config.yaml + env 加载
│   ├── llm.py                   #   ChatOpenAI 工厂（DeepSeek API）
│   ├── compression.py           #   TokenBudget 感知的上下文压缩
│   ├── memory.py                #   三层记忆：工作 / 情景 / 语义
│   ├── retry.py                 #   Tenacity 指数退避 + 瞬时/致命分类
│   ├── prompts/                 #   节点 prompt 模板（.txt）+ few-shot 示例（.jsonl）
│   └── nodes/                   #   8 个节点实现
│       ├── query_router.py      #     关键词启发式 + LLM 兜底 → 路由检索 vs 直答
│       ├── planner.py           #     retrieval_scout + planner（两段式 query_class）
│       ├── executor.py          #     DAG 调度 + $tN.value 解析 + tool dispatch + fact 抽取
│       ├── plan_critic.py       #     按需 plan 修订（信号词 / failed todo 触发）
│       ├── verifier.py          #     结构化 MissingFact + 3 实例并行投票
│       ├── remediation.py       #     根因分派 → 显式 tool_calls + 预算扣减
│       └── synthesizer.py       #     引用答案 + 流式 + 正则引用解析
├── tools/                       # 工具层 — registry + 4 个内置工具
│   ├── registry.py              #     ToolSpec / REGISTRY / dispatch()
│   ├── colpali_tool.py          #     ColQwen2 检索（in-memory / Qdrant / remote）
│   ├── vlm_tool.py              #     VLM 页面阅读（OpenAI-compat）+ SQLite 缓存
│   ├── calculator.py            #     AST 受限安全表达式求值器
│   ├── fact_extractor.py        #     正则结构化 fact 抽取
│   ├── disambiguate.py          #     口径消歧工具
│   └── vlm_cache.py             #     (image_path, instruction) → 缓存 VLM 输出
├── skills/                      # 技能系统 — 可复用 Tool+Prompt+Strategy 能力单元
│   ├── registry.py              #     YAML 技能加载 + 触发关键词 O(1) 匹配
│   ├── single_fact.yaml         #     单事实查询技能配置
│   ├── multi_step_calc.yaml     #     多步计算技能配置
│   ├── cross_doc_compare.yaml   #     跨文档对比技能配置
│   └── trend_analysis.yaml      #     趋势分析技能配置
├── ingestion/                   # 离线数据管线
│   ├── pdf_to_pages.py          #     PDF → 每页 PNG
│   ├── build_index.py           #     ColQwen2 编码 → .pt 多向量索引
│   ├── model_loader.py          #     ColQwen2 模型加载 + 编码共享逻辑
│   ├── push_to_qdrant.py        #     .pt → Qdrant upsert（幂等）
│   └── upload.py                #     用户上传流水线（save → convert → encode → index）
├── services/                    # 模型服务
│   └── colqwen_server.py        #     Litserve ColQwen2 GPU 服务
├── backend/                     # FastAPI 后端
│   ├── server.py                #     POST /query SSE + CRUD + upload + conv_facts
│   ├── storage.py               #     SQLite（conversations / messages / documents / conv_facts）
│   └── schemas.py               #     API 请求/响应模型
├── app/                         # 前端
│   ├── chainlit_app.py          #     Chainlit UI（SSE 消费 + Step 渲染 + TodoList）
│   └── data_layer.py            #     Chainlit DataLayer → 后端 SQLite
├── eval/                        # 评测
│   ├── queries.yaml             #     评测 QA 对（当前 3 题，目标 30 题）
│   └── run_eval.py              #     评测运行脚本
├── config.yaml                  # 全局配置（模型 / 检索 / 服务）
├── docker-compose.yml           # Qdrant 容器
└── requirements.txt
```

---

## 5. 快速开始

### 环境准备

- Python 3.10+
- CUDA 12.1（本地 ColQwen2 推理需要；使用远程服务则不需要）
- Poppler（`pdf2image` 依赖：Linux `apt install poppler-utils`，macOS `brew install poppler`）

### 安装配置

```bash
# 1. 环境
conda create -n findoc python=3.10 -y && conda activate findoc
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. 配置 API key
cp .env.example .env
# 编辑 .env：设置 DEEPSEEK_API_KEY 和 QWEN_API_KEY

# 3. 构建索引（首次运行）
python -m ingestion.pdf_to_pages --only "贵州茅台2023" --max_pages 5
python -m ingestion.build_index --only moutai_2023

# 4. 端到端冒烟（无 key 走 stub fallback）
python -m agent.graph
```

### 启动服务

```bash
# （可选）启动 Qdrant 用于生产环境多向量检索
docker compose up -d qdrant
python -m ingestion.push_to_qdrant

# （可选）在 GPU 机器上启动 ColQwen 服务
python -m services.colqwen_server --port 8000

# 启动后端
PYTHONPATH=. uvicorn backend.server:app --host 0.0.0.0 --port 8001 &

# 启动前端
chainlit run app/chainlit_app.py -w
```

### 配置组合

编辑 `config.yaml` 选择部署模式：

```yaml
retriever:
  backbone: colqwen2          # 或 colpali
  backend: in_memory          # 或 qdrant
  top_k: 5
services:
  colqwen_url: ""             # 或 http://gpu-host:8000
agent:
  max_reflexion_iter: 3
```

---

## 6. 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| **Agent 编排** | LangGraph | 显式状态机，节点 I/O 可观测，条件边控制 reflexion |
| **LLM（文本推理）** | DeepSeek V4 Flash | OpenAI 兼容 API；Planner / Verifier / Synthesizer 共享工厂 |
| **视觉检索** | ColQwen2 | 多向量 + MaxSim；用户 LoRA 基于此 backbone；RTX 3060 6GB bf16 可跑 |
| **VLM（页面阅读）** | Qwen VLM（DashScope） | 页面图 → 结构化中文金融文本 |
| **多向量存储** | Qdrant 1.13 | 原生 `MultiVectorConfig(comparator=MAX_SIM)`；`Distance.DOT` 与 einsum 一致 |
| **模型服务** | Litserve | vLLM 不支持多向量 encoder；Litserve Python 原生 + GPU batching |
| **后端** | FastAPI + SSE | 单向流式推送，无需 WebSocket 开销 |
| **前端** | Chainlit | Python 原生，LangGraph 一等公民，agent/tools/ingestion 零修改 |
| **错误恢复** | Tenacity | 指数退避；区分瞬时重试 vs 致命立即失败 |

---

## 7. 开发路线图

| 阶段 | 状态 | 内容 |
|---|---|---|
| P1–P4 | ✅ | 骨架：目录布局、AgentState、LangGraph 装配、节点/工具 stub、CLI 冒烟 |
| P5–P6 | ⏳ | Chainlit 前端 + 评测集（30 题 QA pairs——当前 3 题） |
| P7–P10 | ✅ | ColQwen2 Litserve 服务化、Qdrant 多向量、SSE 进度推送 |
| P11–P18 | ✅ | VLM 并发、VLM 缓存、对话历史、文档上传、流式输出、自动标题、知识库面板 |
| P19–P25 | ✅ | 结构化 Verifier、差异化修复、Tool Registry、检索感知 Planner、Grounding 审计、结构化 fact 抽取、跨轮记忆 |
| **P26** | ✅ | **错误恢复：** 全链路重试 + 超时 + 结构化 `error_log` |
| **P27** | ✅ | **上下文压缩：** 结构化摘要 + TokenBudget 管理 |
| **P28** | ✅ | **记忆系统升级：** 三层架构（工作/情景/语义）+ cosine 语义匹配 |
| **P29** | ✅ | **DAG 执行：** LLMCompiler 风格 `$tN.value` 数据流 + plan_critic 按需修订 |
| **P30** | ✅ | **动态提示词：** query_class 驱动 prompt 变体 + few-shot 注入（含 3 个 DAG 示例） |
| **P31** | ✅ | **多 Agent：** 3 实例并行验证（strict/base/numeric 多数表决） |
| **P32** | ✅ | **技能系统：** YAML 配置 Tool+Prompt+Strategy 能力单元 + 触发关键词匹配 |

详细工程决策与变更日志见 [DEVLOG.md](./DEVLOG.md)，核心概念深度解析见 [LEARNLOG.MD](./LEARNLOG.MD)。

---

## 8. 参考文献

- **ColPali:** Faysse et al., *Efficient Document Retrieval with Vision Language Models*, 2024
- **ColQwen2:** *Exploring Visual Language Models for Document Retrieval*, 2025 — [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1)
- **LLMCompiler:** Kim et al., *An LLM Compiler for Parallel Function Calling*, ICML 2024
- **Reflexion:** Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS 2023
- **MaxSim（后期交互）:** Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*, SIGIR 2020
- **LangGraph:** [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- **Qdrant Multivector:** [qdrant.tech/documentation/concepts/vectors/#multivector](https://qdrant.tech/documentation/concepts/vectors/#multivector)

---

<p align="center">
  <sub>语言: <a href="./README.md">English</a> | <a href="./README.zh-CN.md">简体中文</a></sub>
</p>
