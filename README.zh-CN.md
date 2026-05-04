# FinDoc Agent

> 面向视觉密集型金融文档（年报、招股书、研报）的多模态 RAG Agent。基于 **ColQwen2** 视觉检索 + **LangGraph** 7 节点状态机，具备结构化根因诊断、差异化反射修复、后置引用/数值审计和跨轮事实复用能力。

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
- **Agent 层**负责任务分解、工具调度、根因诊断、差异化修复、后置引用/数值审计——把"召回的页面"转化为"带引用且经过校验的结构化回答"。

### 当前能力（P25 阶段）

| 能力 | 说明 |
|---|---|
| **多文档问答** | 跨 14+ 已索引年报自动路由查询 |
| **跨公司对比** | 支持口径消歧的跨公司指标对比 |
| **数值计算** | AST 安全表达式求值（如 `(1505.6 + 1420.3) / 2`） |
| **结构化反射** | 四类根因诊断：检索遗漏 / 阅读遗漏 / 查询歧义 / 跨页不一致 |
| **后置审计** | 引用真实性校验 + 数值模糊匹配（±0.1%）；未验证项剥离 + 置信度标识 |
| **跨轮记忆** | 结构化事实 (实体, 期间, 指标, 数值, 单位) 跨对话轮次复用；追问跳过检索 |
| **流式输出** | Synthesizer token-by-token SSE 实时推送 |
| **用户上传文档** | PDF/图片上传 → 自动编码 → Qdrant 索引 → 立即可查询 |
| **知识库管理** | 面板 UI：列表 / 删除 / 重新索引 / 封面预览 |

---

## 2. 系统架构

### 2.1 Agent 工作流（7 节点状态机）

```
用户问题
    │
    ▼
┌──────────────────┐  全库 MaxSim 预检索 → top-3 候选文档 + 相关性分数
│ retrieval_scout  │  (P22)
└────────┬─────────┘
         │
         ▼
┌──────────┐         分解问题 → 有序 SubTask 列表
│ Planner  │  ──►    每个 SubTask: sub_query / target_doc / tool_calls / query_class
└──────────┘         输入含 candidate_docs + 可用工具列表
    │
    ▼
┌──────────┐         tool_calls → registry dispatch (P21)
│ Executor │  ──►    回退 expected_output_schema 旧路径
└──────────┘         VLM 输出 → fact_extractor 结构化 (P24)
    │                检索前先查 known_facts 跨轮缓存 (P25)
    ▼
┌──────────┐         结构化 missing_facts（含 root_cause + confidence）
│ Verifier │  ──►    早停：无新事实 → 强制 synthesizer (P19)
└──────────┘
    │
    ├─ 充分 ────────────────────────┐
    │                                ▼
    ├─ 不充分 ──► ┌──────────┐  按根因分派 4 条修复路径 (P20):
    │             │Remediation│  retrieval_miss → 放宽 top_k 重检索
    │             └─────┬─────┘  reading_miss → 精炼指令重读同批页
    │                   │        ambiguous_query → 改写自包含查询
    │                   ▼        inconsistency → 口径消歧
    │               Executor     budget 耗尽 → 强制 fallthrough
    │                   │
    └───────────────────┘
                        ▼
                 ┌─────────────┐
                 │ Synthesizer │  汇总事实 → [doc_id p.X] 引用答案 (流式)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐  引用反向校验 + 数值模糊匹配
                 │  Grounding  │  剥离未验证引用 + 置信度 banner (P23)
                 └─────────────┘
```

### 2.2 部署架构

```
┌─ uvicorn backend.server:app (port 8001) ──────┐
│  FastAPI + SSE                                  │
│  POST /api/v1/query     → SSE 流（7 节点进度） │
│  GET  /api/v1/documents → 已索引文档列表        │
│  GET  /api/v1/health                            │
│  startup → 预加载 ColQwen2 + 索引               │
│  per-query → load/save conv_facts (P25)         │
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
│  消费 SSE：event=node|status|token|done        │
│  动态 Step + 引用 inline Image + 置信度 banner │
└────────────────────────────────────────────────┘
```

**关键边界：** `agent/`、`tools/`、`ingestion/` 是业务核心，由后端直接 import；前端仅通过 HTTP/SSE 调用，不 import 任何 agent 代码。

### 2.3 节点职责

| 节点 | 职责 | 新增于 |
|---|---|---|
| `retrieval_scout` | 预检索全库，返回 top-3 候选文档 + 相关性分数，让 planner 做知情决策 | P22 |
| `planner` | 分解 `question → [SubTask...]`，含 `tool_calls` / `target_doc` / `query_class` | P1 |
| `executor` | tool_calls 优先走 registry dispatch；回退旧版 schema 路由；VLM 并发读取 | P1 + P21 |
| `verifier` | 结构化 sufficiency + consistency 判断 → `MissingFact[]`（含 root_cause） | P1 + P19 |
| `remediation` | 按 root_cause 分派 4 条修复策略，budget 扣减控制 | P20 |
| `synthesizer` | 汇总事实 → 带 `[doc_id p.N]` 引用答案，流式 token 输出 | P1 + P16 |
| `grounding` | 后置审计：逐条校验引用真实性 + 数值与 evidence 一致性，剥离未验证项 | P23 |

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

LangChain 的 AgentExecutor 是黑箱循环。LangGraph 是显式状态机——每个节点的 I/O 可观测，reflexion 循环通过条件边控制，整个拓扑在 `graph.py` 里仅 20 行装配代码。

### 3.2 为什么用 ColQwen2 做视觉检索？

传统 RAG：`PDF → OCR → 文本分块 → 单向量 embedding → 语义搜索`。金融文档表格密集，OCR 出错率高，单向量压缩丢失空间布局信息。

ColQwen2 将每页图片编码为**多向量**（每 patch 一个向量，每页约 1024 个 × 128 维）。检索用 **MaxSim**——query 的每个 token 找文档中最相似的 patch，再求和：

```
score(q, d) = Σ_{i ∈ q_tokens} max_{j ∈ d_tokens} ⟨q_i, d_j⟩
```

表格结构、图表布局、数字位置信息完整保留。ViDoRe benchmark：nDCG@5 ~89%（ColPali ~81%）。2B 基座在 RTX 3060 6GB bf16 推理不会 OOM。

### 3.3 为什么必须用 Qdrant？

标准向量库（Chroma、Pinecone、pgvector）只支持单向量存一条记录。一份 200 页年报产生约 20 万个向量——Qdrant 1.10+ 是少数原生支持 `MultiVectorConfig(comparator=MAX_SIM)` 的数据库。使用 `Distance.DOT`（不用 COSINE）保证与 Python einsum 点积结果完全一致。

### 3.4 结构化 Reflexion（P19–P20）

Verifier 不再输出模糊的"还需要更多信息"字符串，而是结构化的 `MissingFact` 列表，每条带 `root_cause` 枚举：

| root_cause | 修复策略 |
|---|---|
| `retrieval_miss` | 放宽 `top_k`，用改写后的 query 重检索 |
| `reading_miss` | 精炼 VLM instruction，重读同一批页（不走 ColQwen） |
| `ambiguous_query` | 改写为完全自包含的查询 |
| `inconsistency` | 触发 `disambiguate_caliber` 对冲突页逐页提取披露口径 |

三重防死循环保护：`max_reflexion_iter=3` + `budget_retrievals=10` + `budget_vlm_calls=20`。

### 3.5 Tool Registry（P21）

工具自描述 `(name, description, params_schema, output_schema)`，planner prompt 自动感知可用工具，executor 纯 dispatch + 输出校验。加新工具只需一行 `register(ToolSpec(...))`——无需改 planner、executor 或 schemas。

| 工具 | 类别 | 功能 |
|---|---|---|
| `retrieve_pages` | retrieval | ColQwen2 多向量 MaxSim 检索 |
| `read_page_with_vlm` | reading | VLM 页面图 → 结构化文本 |
| `calculate` | compute | AST 受限安全数值求值 |
| `disambiguate_caliber` | resolution | 跨页数值冲突 → VLM 口径提取 |

### 3.6 Grounding 后置审计（P23）

Synthesizer 输出后，纯规则校验（正则 + 集合查，零 LLM 调用）：
- **引用校验：** 答案中每个 `[doc_id p.N]` 必须在 `extracted_facts` 中存在
- **数值校验：** 每个数值+单位必须与 evidence 模糊匹配（±0.1%）
- **置信度 banner：** 全通过无标识，部分失配 ⚠，严重失配 🛑

### 3.7 跨轮事实记忆（P24–P25）

每次 VLM 读取后，`fact_extractor`（纯 regex，零 LLM）从中文金融文本中抽取 `(实体, 期间, 指标, 数值, 单位)`。结构化 facts 落 SQLite `conv_facts` 表。下轮追问时 executor 先查 `known_facts`——若 `(茅台, 2023, 营收)` 已命中，直接跳过 retrieval+VLM。连续追问场景预期减少 40–60% 检索调用。

---

## 4. 项目结构

```
findoc-agent/
├── agent/                       # Agent 核心 — 7 节点 LangGraph 状态机
│   ├── graph.py                 #   build_graph() / compile_graph() 图定义
│   ├── state.py                 #   AgentState TypedDict + Fact / SubTask / PageHit
│   ├── schemas.py               #   LLM 结构化输出 schema（PlannerOutput, VerifierOutput）
│   ├── config.py                #   config.yaml + env 加载
│   ├── llm.py                   #   ChatOpenAI 工厂（DeepSeek API）
│   ├── prompts/                 #   节点 prompt 模板（.txt）
│   └── nodes/                   #   7 个节点实现
│       ├── planner.py           #     retrieval_scout + planner (P22)
│       ├── executor.py          #     tool dispatch + VLM 读取 + fact 抽取 (P21/P24/P25)
│       ├── verifier.py          #     充分性 + 一致性 + 结构化 missing_facts (P19)
│       ├── remediation.py       #     root_cause 分派 → 4 条修复策略 (P20)
│       ├── synthesizer.py       #     带引用答案 + 流式 token 输出 (P16)
│       └── grounding.py         #     引用 + 数值后置审计 (P23)
├── tools/                       # 工具层 — registry + 4 个内置工具
│   ├── registry.py              #     ToolSpec / REGISTRY / dispatch() (P21)
│   ├── colpali_tool.py          #     ColQwen2 检索（in-memory / Qdrant / remote）
│   ├── vlm_tool.py              #     VLM 页面阅读（OpenAI-compat）+ SQLite 缓存
│   ├── calculator.py            #     AST 受限安全表达式求值器
│   ├── fact_extractor.py        #     正则结构化 fact 抽取 (P24)
│   ├── disambiguate.py          #     口径消歧工具 (P20)
│   └── vlm_cache.py             #     (image_path, instruction) → 缓存 VLM 输出
├── ingestion/                   # 离线数据管线
│   ├── pdf_to_pages.py          #     PDF → 每页 PNG
│   ├── build_index.py           #     ColQwen2 编码 → .pt 多向量索引
│   ├── model_loader.py          #     ColQwen2 模型加载 + 编码共享逻辑
│   ├── push_to_qdrant.py        #     .pt → Qdrant upsert（幂等）
│   └── upload.py                #     用户上传流水线（save → convert → encode → index）
├── services/                    # 模型服务
│   └── colqwen_server.py        #     Litserve ColQwen2 GPU 服务
├── backend/                     # FastAPI 后端
│   ├── server.py                #     POST /query SSE + CRUD + upload + conv_facts (P25)
│   ├── storage.py               #     SQLite（conversations / messages / documents / conv_facts）
│   └── schemas.py               #     API 请求/响应模型
├── app/                         # 前端
│   ├── chainlit_app.py          #     Chainlit UI（SSE 消费 + Step 渲染 + 置信度 banner）
│   └── data_layer.py            #     Chainlit DataLayer → 后端 SQLite
├── eval/                        # 评测
│   ├── qa_dataset.jsonl         #     评测 QA 对（计划 30 题，当前示例）
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

---

## 7. 开发路线图

| 阶段 | 状态 | 内容 |
|---|---|---|
| P1–P4 | ✅ | 骨架：目录布局、AgentState、LangGraph 装配、节点/工具 stub、CLI 冒烟 |
| P5–P6 | ⏳ | Chainlit 前端 + 评测集（30 题 QA pairs） |
| P7–P10 | ✅ | ColQwen2 Litserve 服务化、Qdrant 多向量、SSE 进度推送 |
| P11–P18 | ✅ | VLM 并发、VLM 缓存、对话历史、文档上传、流式输出、自动标题、知识库面板 |
| P19–P25 | ✅ | 结构化 Verifier、差异化修复、Tool Registry、检索感知 Planner、Grounding 审计、结构化 fact 抽取、跨轮记忆 |
| **P26** | ⏳ | **错误恢复：** 全链路重试 + 超时 + error_log |
| **P27** | ⏳ | **上下文压缩：** 结构化摘要替代暴力截断 + TokenBudget 管理 |
| **P28** | ⏳ | **记忆系统升级：** 语义匹配 + 三层架构（Working/Episodic/Semantic）+ 反馈闭环 |
| **P29** | ⏳ | **任务系统：** DAG 依赖图 + 同层并发 + plan_critic 按需修订 |
| **P30** | ⏳ | **动态提示词：** query_class 驱动 prompt 变体 + few-shot 注入 |
| **P31** | ⏳ | **多 Agent：** Parallel Verification 多数表决 + Supervisor/Specialist 路由 |
| **P32** | ⏳ | **技能系统：** Tool+Prompt+Strategy 可复用能力单元 |

详细工程决策见 [DEVLOG.md](./DEVLOG.md)，核心概念深度解析见 [LEARNLOG.MD](./LEARNLOG.MD)。

---

## 8. 参考文献

- **ColPali:** Faysse et al., *Efficient Document Retrieval with Vision Language Models*, 2024
- **ColQwen2:** *Exploring Visual Language Models for Document Retrieval*, 2025 — [vidore/colqwen2-v0.1](https://huggingface.co/vidore/colqwen2-v0.1)
- **Reflexion:** Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS 2023
- **MaxSim（后期交互）:** Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT*, SIGIR 2020
- **LangGraph:** [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
- **Qdrant Multivector:** [qdrant.tech/documentation/concepts/vectors/#multivector](https://qdrant.tech/documentation/concepts/vectors/#multivector)

---

<p align="center">
  <sub>语言: <a href="./README.md">English</a> | <a href="./README.zh-CN.md">简体中文</a></sub>
</p>
