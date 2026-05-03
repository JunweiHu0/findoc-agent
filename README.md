# FinDoc Agent

> 面向视觉密集型金融文档（年报、招股书、研报）的多模态 RAG Agent。以 **ColQwen2** 为视觉检索引擎，基于 **LangGraph** 构建 Planner–Executor–Verifier–Synthesizer 四节点状态机，支持跨文档对比、表格/图表解读与带反思机制的多轮检索。前端 Chainlit + FastAPI SSE，后端服务化 ColQwen2 / Qdrant 多向量。

[![Python](https://img.shields.io/badge/python-3.10+-green)]() [![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## 1. 项目背景与动机

金融年报这类文档有两个特点让传统文本 RAG 表现很差：

1. **版式复杂、信息密集**：财务报表、趋势图、组织架构图、合并报表附注，文本 chunk 后语义被切碎，OCR 还原结构损耗大。
2. **跨页跨文档推理需求强**：用户问"对比 A、B 两家公司近三年毛利率趋势"，本质是一个需要多步检索 + 数值计算 + 对比生成的任务，单轮 RAG 无法覆盖。

本项目的核心思路：**用 ColQwen2 视觉检索解决"看得见"的问题，用 Agent 状态机解决"想得清"的问题**。

- ColQwen2 直接以页面图像为检索单位，多向量 + MaxSim 后期交互，绕过 OCR/版面解析，对表格、图表、版式复杂页面的召回显著优于文本检索；
- Agent 层负责任务规划、工具调用、反思与重检索，把"召回的页面"转化为"带引用的结构化回答"。

## 2. 系统架构

### 2.1 Agent 工作流

```
用户问题
    │
    ▼
┌──────────┐     拆解为有序子任务（JSON plan）
│ Planner  │ ──► 每个 SubTask: query / target_doc / expected_schema
└──────────┘
    │
    ▼
┌──────────┐     text/table → ColQwen2 检索 → VLM 读页 → 抽取事实
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

整套流程用 **LangGraph** 实现为显式状态机，每个节点的输入/输出/状态变更都可观测、可回放。

### 2.2 部署架构（P10 之后）

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

## 3. 关键设计

### 3.1 Planning：轻量级 Decomposition Planner

不采用 ReAct 式的"边想边做"，而是在入口先做一次显式任务分解：

- **输入**：用户原问题 + 文档元信息（公司、年份、可用文件列表）
- **输出**：一个有序的子任务列表，每个子任务带 `target_doc`、`sub_query`、`expected_output_schema`
- **为什么这么做**：金融问答的子任务边界是清晰的（"取数 → 计算 → 对比"），先规划再执行比 ReAct 在该场景下更稳定，token 开销也更低。
- **降级策略**：若 Planner 输出无法解析为合法 JSON，自动回退为单步执行（把原问题直接塞进 Executor），保证鲁棒性。

### 3.2 Tool Use：三个聚焦的工具

| Tool | 职责 | 实现 |
|---|---|---|
| `colpali_retrieve(query, top_k, doc_filter)` | 视觉检索，返回 top-k 页面（图像 + 元信息） | ColQwen2 多向量 + MaxSim（in-memory 或 Qdrant 服务端）|
| `vlm_read_page(page_image, instruction)` | 对召回页做结构化抽取（数字、趋势、表头） | Qwen VLM（DashScope OpenAI-compat）|
| `calculator(expr)` | 安全数值计算（毛利率、同比、环比） | 基于 AST 的受限 eval，禁用任意函数调用 |

设计原则：**Tool 数量保持在 ≤ 3，每个 Tool 边界正交**。

### 3.3 Reflexion：基于充分性 + 一致性的双重校验

Verifier 节点是本项目的核心差异点。每轮 Executor 执行完后，Verifier 检查两件事：

1. **充分性（Sufficiency）**：当前 working memory 里的证据是否足以回答原问题？
   - 用 LLM 做结构化判断，输出 `{is_sufficient: bool, missing_info: str}`
   - 不充分时，把 `missing_info` 作为新的 sub_query 追加到 plan，下轮 Executor 直接消费
2. **一致性（Consistency）**：多页召回到的数字是否一致？跨文档对比时的口径是否一致（合并报表 vs 母公司报表）？
   - 不一致时，触发"溯源"——让 Executor 重新读取原页定位差异。

**防死循环机制**：`max_reflexion_iterations = 3`，超过后强制进入 Synthesizer 并在最终答案中标注"低置信度"。

参考 Reflexion (Shinn et al., 2023)，但限定为"客观可判定"的两类校验，不做通用 self-critique（容易让 LLM 自我说服）。

### 3.4 引用与可追溯性

最终回答中每一个事实性陈述都会附带 `[doc_id p.X]` 引用，前端以 inline 图片展示召回页截图。这一点在金融场景是合规级别的硬需求。

### 3.5 进度推送（SSE）

LangGraph `astream` 在 Executor 内部被模型加载/VLM 调用阻塞，无法从主循环推中间事件。解法：

- 工具侧（colpali_tool / vlm_tool）：模块级回调在长操作点埋点
- 后端侧：Agent 放后台线程，主 async 循环每 300ms 轮询节点队列 + 进度队列，交替 yield `event: node` 和 `event: status`
- 前端：进度 Step 实时更新（`⏳ VLM 正在读取 p009.png...`），节点完成后展开结构化 markdown

## 4. 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| Agent 编排 | **LangGraph** | 状态机显式，节点可观测，便于实现 reflexion 循环 |
| LLM（文本推理） | **DeepSeek V4 Flash** | OpenAI-Compatible API；Planner / Verifier / Synthesizer 共享 |
| 视觉检索 | **ColQwen2**（vidore/colqwen2-v0.1） | 多向量 + MaxSim 后期交互；用户 LoRA 基于此 backbone；2B 基座适配 RTX 3060 6GB |
| VLM（页面阅读） | **Qwen VLM**（DashScope） | 页面图 → 结构化文本抽取 |
| 多向量存储 | **Qdrant 1.13** | 原生 MultiVectorConfig + MaxSim comparator；单向量库（Chroma/pgvector）不支持 |
| 模型服务 | **Litserve**（Lightning AI） | vLLM 不支持多向量 encoder；Litserve Python 原生 + GPU batching |
| 后端 | **FastAPI** + SSE | 单向流（planner→executor→verifier→synthesizer），无双向需求 |
| 前端 | **Chainlit** | Python 原生、LangGraph 一等公民；amber 暖色主题 |

### 四种部署组合（config 矩阵）

| `services.colqwen_url` | `retriever.backend` | 含义 |
|---|---|---|
| `""` | `in_memory` | 默认开发模式：本地模型 + 本地 .pt MaxSim |
| `http://localhost:8000` | `in_memory` | ColQwen 服务化 + 本地 MaxSim |
| `""` | `qdrant` | 本地编码 + Qdrant 服务端 MaxSim |
| `http://localhost:8000` | `qdrant` | 完全分布式（推荐生产） |

任何 Qdrant 异常自动 fallback 到 in-memory 检索。

## 5. 评测

自建评测集（`eval/qa_dataset.jsonl`），分三类：

- **L1 单页事实型**：可直接从一页定位答案
- **L2 跨页/表格图表型**：需要解读表格或图表
- **L3 跨文档对比型**：需要规划 + 多次检索

> 当前评测集仅含示例条目，P6 计划扩展到 30 题（L1×10 + L2×10 + L3×10）并加入 baseline 对比。评测脚本见 `eval/run_eval.py`。

## 6. 项目结构

```
findoc-agent/
├── agent/                       # Agent 核心（LangGraph 状态机）
│   ├── graph.py                 # compile_graph() 图定义
│   ├── state.py                 # AgentState TypedDict
│   ├── schemas.py               # Pydantic schema（SubTask / ReflexionResult）
│   ├── config.py                # config.yaml + env 加载
│   ├── llm.py                   # ChatOpenAI 工厂（DeepSeek）
│   ├── nodes/
│   │   ├── planner.py           # 任务分解
│   │   ├── executor.py          # 按 schema 路由执行
│   │   ├── verifier.py          # Reflexion 充分性+一致性校验
│   │   └── synthesizer.py       # 汇总 + 引用生成
│   └── prompts/                 # 节点 prompt 模板（.txt）
├── tools/                       # 工具层（被 Executor 调用）
│   ├── colpali_tool.py          # ColQwen2 检索（in-memory / Qdrant / remote）
│   ├── vlm_tool.py              # VLM 页面阅读（OpenAI-compat）
│   └── calculator.py            # AST 受限安全求值
├── ingestion/                   # 离线数据处理
│   ├── pdf_to_pages.py          # PDF → PNG 页面
│   ├── build_index.py           # ColQwen2 编码 → .pt 多向量索引
│   ├── model_loader.py          # ColQwen2 模型加载 + 编码共享逻辑
│   └── push_to_qdrant.py        # .pt → Qdrant upsert（幂等）
├── services/                    # 模型服务
│   └── colqwen_server.py        # Litserve ColQwen2 服务（GPU）
├── backend/                     # FastAPI 后端
│   ├── server.py                # POST /query SSE + startup 预加载
│   └── schemas.py               # API 请求/响应模型
├── app/                         # 前端
│   └── chainlit_app.py          # Chainlit UI（SSE 消费 + Step 渲染）
├── eval/
│   ├── qa_dataset.jsonl         # 评测 QA 对
│   └── run_eval.py              # 评测脚本
├── config.yaml                  # 全局配置（模型/检索/服务/后端）
├── docker-compose.yml           # Qdrant 容器
└── requirements.txt
```

## 7. 快速开始

```bash
# 1. 环境
conda create -n findoc python=3.10 && conda activate findoc
pip install -r requirements.txt

# 2. 配置 API key
cp .env.example .env
# 编辑 .env：填入 DEEPSEEK_API_KEY 和 QWEN_API_KEY

# 3. 构建索引（首次运行）
python -m ingestion.pdf_to_pages --only "贵州茅台2023" --max_pages 5
python -m ingestion.build_index --only moutai_2023

# 4. 端到端冒烟（无 key 走 stub fallback）
python -m agent.graph

# 5. 启动 Qdrant（可选，生产模式）
docker compose up -d qdrant
python -m ingestion.push_to_qdrant

# 6. 启动 ColQwen 服务（可选，GPU 机器上）
python -m services.colqwen_server --port 8000

# 7. 启动后端 + 前端
PYTHONPATH=. uvicorn backend.server:app --host 0.0.0.0 --port 8001 &
chainlit run app/chainlit_app.py -w
```

## 8. 关键设计决策

1. **为什么不直接 fork Dify / FinRobot？** 两者目标都是平台化，与本项目"垂直领域 + 视觉检索深度集成"的定位不符；从零搭建反而更易讲清每一行代码的设计意图。
2. **为什么 Planner 用 decomposition 而非 ReAct？** 金融问答的子任务边界清晰，先规划再执行更稳定且 token 更省；ReAct 在该场景下容易陷入"反复检索同一页"。
3. **为什么 Verifier 只校验充分性 + 一致性？** 通用 self-critique 容易让 LLM 自我说服；限定为"客观可判定"的两类校验更可靠。
4. **为什么 ColQwen2 而非 ColPali？** 用户自微调 LoRA 基于 ColQwen2；ColQwen2 在 ViDoRe 上 nDCG@5 ~89%（ColPali ~81%），密集表格的动态分辨率支持更好，2B 基座在 RTX 3060 6GB bf16 推理不会 OOM。
5. **为什么选 Chainlit 而非 Gradio？** Gradio v1 已弃维；Chainlit Python 原生、对 LangGraph 一等公民，`agent/` `tools/` `ingestion/` 三个核心目录一行不动即可接入。
6. **为什么引入 Qdrant？** 单向量库（Chroma/pgvector）不支持 MaxSim 多向量检索；Qdrant 1.10+ 的 MultiVectorConfig 是 ColPali/ColQwen 落地的事实标准。本地 .pt + 内存 MaxSim 在开发阶段仍可用，Qdrant 为横向扩展预留。

## 9. 局限性与后续工作

当前进度 P10（前后端分离 + SSE 进度推送），剩余计划见 [DEVLOG.md](./DEVLOG.md) §5：

- **P11** VLM 并行化（top-k 串行→并发，10s→2s）
- **P12** VLM 输出缓存（SQLite，命中率预计 30–50%）
- **P13** 历史对话栏（SQLite 持久化 + 左侧栏）
- **P14** 用户上传 PDF/图片自建知识库
- **P15** 多轮上下文（chat_history 接入 Planner）
- **P16** Synthesizer 流式输出（token-by-token）
- **P17** 对话标题自动生成
- **P18** 知识库管理面板

已知硬伤：零测试覆盖、无重试/超时机制、评测集仅示例条目、VLM 串行调用。详见 [BUGS.md](./BUGS.md) 和 [LEARNLOG.MD](./LEARNLOG.MD) §八。

## 10. References

- ColPali: Faysse et al., *Efficient Document Retrieval with Vision Language Models*, 2024
- ColQwen2: *Exploring Visual Language Models for Document Retrieval*, 2025
- Reflexion: Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS 2023
- LangGraph: https://langchain-ai.github.io/langgraph/
- Qdrant Multivector: https://qdrant.tech/documentation/concepts/vectors/#multivector
