# FinDoc Agent 学习笔记

## 项目概述

FinDoc Agent 是一个面向**金融文档（年报、招股书、研究报告）的多模态 RAG Agent**，专门处理中文金融文档。核心思路：绕过传统 OCR + 文本分块，直接用**视觉检索**定位页面，再用 VLM 读取内容，最后通过 LLM 合成答案。

---

## 整体工作流

```
用户问题
    │
    ▼
┌──────────┐     ┌─────────────────────────┐
│ Planner  │ ──► │ 拆解为子任务 (JSON plan)  │
│ DeepSeek │     └──────────┬──────────────┘
└──────────┘                │
                            ▼
                 ┌──────────────────────┐
                 │     Executor          │
                 │  ├─ text/table 子任务  │──► ColPali 检索 → VLM 读页
                 │  └─ number 子任务 ────│──► Calculator 安全求值
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
       ┌─────────│     Verifier          │
       │         │  DeepSeek             │
       │         │  证据充分且一致?        │
       │ 不充分   └──────────┬───────────┘
       │ (追加子任务)         │ 充分
       └─────────────────────┘
                            ▼
                 ┌──────────────────────┐
                 │    Synthesizer        │
                 │   DeepSeek            │
                 │  生成最终答案 + 引用    │
                 └──────────────────────┘
```

---

## 四个 Agent 节点

### 1. Planner（规划器）
- 接收用户问题，调用 DeepSeek API 将问题拆解为有序的 SubTask 列表
- 每个子任务指定：搜索查询词、目标文档（可选）、期望输出类型（text / number / table）
- 例如 "对比茅台和宁德时代的毛利率" → 两个检索子任务 + 一个计算子任务

### 2. Executor（执行器）
- **text/table 子任务**：ColPali/ColQwen2 视觉检索 → VLM 逐页阅读 → 提取结构化事实
- **number 子任务**：安全 AST 计算器求值算术表达式

### 3. Verifier（验证器）
- 检查证据是否足以回答问题，跨页面数字是否一致
- 不足或不一致 → 追加子任务，重新进入 Executor
- 最多反射 3 次，之后强制进入 Synthesizer

### 4. Synthesizer（合成器）
- 汇总所有事实和计算值，调用 LLM 生成最终答案
- 每个事实声明必须附 `[doc_id p.X]` 内联引用
- 不引入证据之外的信息

---

## 模型体系（3 类模型，各司其职）

| 模型 | 调用方式 | 用途 |
|------|---------|------|
| **DeepSeek V4 Flash** | API 远程 | Planner / Verifier / Synthesizer 推理 |
| **ColQwen2** | 本地加载 | 视觉检索 backbone，MaxSim 多向量匹配 |
| **Qwen VLM** | API 远程 (DashScope) | 页面图片 → 结构化文本 |

### 1. DeepSeek V4 Flash（文本 LLM）

调用方式：API 远程调用（`DEEPSEEK_API_KEY`）

用途：驱动三个 Agent 节点的推理：
- Planner — 把用户问题拆解成有序子任务
- Verifier — 判断已收集的证据是否充分、一致
- Synthesizer — 汇总事实，生成带引用的最终答案

这是纯文本模型，不做任何视觉处理。

### 2. ColQwen2（视觉检索 backbone）

调用方式：本地加载（`colpali_engine` Python 包，`models/colqwen2-v0.1/`）

用途：将页面图片和文本查询分别编码为多向量嵌入，然后做 MaxSim 相似度匹配，找出与查询最相关的页面。

**ColQwen 是整个系统的核心**——它解决了金融文档检索的根本难题。传统 RAG 的做法是 PDF → OCR → 文本分块 → 文本 embedding → 语义搜索，但金融文档（年报、招股书）有大量表格、图表、数字密集的排版，OCR 容易出错，文本分块会破坏表格的语义结构。

ColQwen 直接用视觉方式理解页面，不需要 OCR：
- 把每页 PDF 渲染为图片，用 ColQwen 的 vision encoder 编码成多向量 embedding（每页多个 token 各有自己的向量）
- 查询时同样把文本编码成多向量
- 用 **MaxSim**（每对 query token × page token 取最大相似度，再求和）计算相关性分数
- 返回最匹配的页面图片

这样表格、图表、数字布局的信息都被完整保留，不会因为 OCR 或分块而丢失精度。

### 3. Qwen 视觉模型（VLM 页面阅读器）

调用方式：API 远程调用（阿里云 DashScope，`QWEN_API_KEY`）

用途：ColQwen 检索出相关页面图片后，需要有人"读"图片里的内容。Qwen 视觉模型接收页面 PNG + 指令，输出结构化的文本描述，比如 "2023 年茅台酒毛利率为 94.12%，见合并利润表"。

### 协作流程

```
用户查询: "茅台2023年毛利率是多少?"
           │
           │  ① 文本查询
           ▼
      ColQwen2 ──────────► 编码查询为多向量
      (本地推理)              │
                              │ MaxSim 匹配
                              ▼
                           返回 top-k 相关页面图片
                              │
                              │ ② 页面图片 + 查询指令
                              ▼
                          Qwen VLM ──────► "毛利率94.12%"
                         (DashScope API)
                              │
                              │ ③ 结构化事实
                              ▼
                          DeepSeek ──────► 最终带引用答案
                          (API)
```

ColQwen 是**检索引擎**（找哪几页有答案），Qwen VLM 是**阅读器**（从图片里读出内容），DeepSeek 是**推理大脑**（规划、验证、合成）。

---

## 离线数据准备

1. `pdf_to_pages.py`：PDF → 逐页 PNG 图片
2. `build_index.py`：ColQwen2 编码每页 → 多向量嵌入保存为 `.pt`，生成 `doc_memory.json`

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 文本 LLM | DeepSeek V4 Flash API |
| 视觉检索 | ColQwen2 (本地，MaxSim) |
| VLM 页面阅读 | Qwen 视觉模型 (DashScope) |
| 编排 | LangGraph StateGraph |
| 前端 | Gradio 三栏界面（v1，已上线）→ Chainlit（v2，规划中，详见末节） |

---

## 工业级差距分析与改进建议

按严重程度排序：**严重**（阻碍上线）> **高**（影响可靠性）> **中**（影响开发效率）

---

### 1. 缓存机制（严重）

**现状**：无任何运行时缓存。

- LLM 调用每次都是全新请求，相同/相似的 plan、verify、synthesize 查询重复消耗 API 费用
- VLM 页面读取无缓存，同一页面图片多次请求会重复调用 API
- 唯一缓存是 `tools/colpali_tool.py:36` 中对 ColQwen 模型/索引的模块级内存驻留，只判断加载与否，没有 TTL、淘汰策略

**建议**：
- **LLM 结果缓存**：引入 Redis，对 Planner/Verifier/Synthesizer 的输出做语义缓存（key = query + 文档元信息的哈希）。高频金融问题（"毛利率是多少"）命中率会很高
- **VLM 页面缓存**：用 `(doc_id, page_num, query_hash)` 做 key，缓存 VLM 读取结果到 Redis 或本地 SQLite。同一页面被多个子任务检索到时直接复用
- **检索结果缓存**：ColPali MaxSim 结果对相同 query 缓存到内存 LRU，避免重复编码
- 考虑引入 `diskcache` 做本地降级方案

### 2. 弹性与重试机制（高）

**现状**：所有节点有一个 try/except 包裹，但失败直接回退到 mock/heuristic 结果，不做任何重试。

- LLM 调用没有 `timeout`、`max_retries` 配置（`agent/llm.py:21-27`）
- VLM API 调用（`tools/vlm_tool.py:65`）无超时设置，OpenAI SDK 默认无限等待
- 无断路器模式：API 连续失败后，每个节点仍独立发起请求
- 无 fallback chain：DeepSeek 挂了不能切到其他 provider

**建议**：
- 引入 `tenacity` 库，为所有外部 API 调用添加指数退避重试（`@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))`）
- `agent/llm.py` 中给 ChatOpenAI 加 `timeout=60`、`max_retries=3`
- VLM 调用用 `httpx.Timeout(30.0, read=60.0)`
- 增加断路器状态机：连续失败 N 次 → 熔断 M 秒 → 半开探测 → 恢复或继续熔断
- 可选：LLM fallback chain（DeepSeek → 阿里通义千问 → 本地模型）

### 3. 可观测性（严重）

**现状**：导入了 `loguru` 但从未配置（无 `logger.add()`、无 JSON 格式化、无日志轮转）。无指标、无链路追踪。

- `agent/config.py:34` 读取了 `LOG_LEVEL` 但只在 Planning 节点日志中用了一下
- 无 LLM 调用延迟、成功率、token 消耗统计
- 无检索延迟、VLM 调用计数监控
- 无 Sentry/同类错误追踪

**建议**：
- **结构化日志**：`logger.add("logs/app_{time}.log", rotation="100 MB", retention="7 days", serialize=True)` 输出 JSON 格式，便于接入 ELK/Loki
- **链路追踪**：接入 LangSmith 或 LangFuse（LangGraph 原生支持，只需设置 `LANGCHAIN_TRACING_V2=true`），可视化每一步的耗时和 token 消耗
- **指标收集**：用 `prometheus_client` 暴露 `/metrics`，记录：
  - `llm_call_duration_seconds`（histogram，按节点分 label）
  - `colpali_retrieve_duration_seconds`
  - `vlm_read_duration_seconds`
  - `reflexion_loop_count`（counter）
  - `api_error_count`（counter）
- **错误追踪**：接入 Sentry `sentry_sdk`，在 `gradio_app.py` 的最外层 except 中上报

### 4. 测试（严重）

**现状**：零个测试文件，无 `tests/` 目录，无 pytest 配置。

- `tools/calculator.py` 的 AST 安全求值器是理想的单元测试目标（需测恶意表达式、边界值）
- `ingestion/pdf_to_pages.py:25` 的 `derive_doc_id` 纯逻辑函数，无测试
- `tools/colpali_tool.py` 的 MaxSim 计算可以造小张量做确定性测试
- 无集成测试验证 agent 图的端到端行为

**建议**：
- 创建 `tests/` 目录，引入 `pytest` + `pytest-cov` + `pytest-mock`
- **L0 单元测试**：calculator（安全评估+边界）、derive_doc_id、schemas 序列化/反序列化
- **L1 集成测试**：mock 外部 API，验证 Planner → Executor → Verifier → Synthesizer 状态流转
- **L2 E2E 测试**：用评测数据集 `eval/qa_dataset.jsonl` 做回归测试，接入 LLM-as-Judge 自动评分
- CI 门禁：`pytest --cov --cov-report=term-missing`

### 5. 配置管理（中）

**现状**：单一 `config.yaml`，无环境区分。API 密钥明文存在 `.env` 文件中。

**建议**：
- 分层配置：`config.yaml`（基础值，可提交）+ `config.prod.yaml`（覆盖值）+ 环境变量覆盖（最高优先级）
- 生产环境密钥：用云 KMS（阿里云 KMS / HashiCorp Vault），不落盘明文
- 或至少用 `python-dotenv` 的 `.env.production` + 运维侧注入环境变量
- 引入 `pydantic-settings` 做配置校验，启动时验证所有必需配置项存在且合法

### 6. 并发与速率限制（严重）

**现状**：无任何并发控制或速率限制。

- `app/gradio_app.py:35` 中 `_app` 是模块级单例，LangGraph `CompiledGraph.stream()` 不是可重入的，多用户同时查询存在竞态条件
- 无 LLM/VLM API 的 token bucket 限速，恶意或 bug 循环可能瞬间打爆 API 配额
- Executor 中对每个检索命中顺序调用 `vlm_read_page`（`agent/nodes/executor.py:39-47`），top_k=5 就是 5 次串行 API 调用

**建议**：
- **修复竞态条件**：每次请求创建独立的 `CompiledGraph` 副本，或确保 graph 本身无共享可变状态。检索器的模型/索引可以用 `threading.Lock` 保护
- **并发 VLM 调用**：用 `concurrent.futures.ThreadPoolExecutor(max_workers=3)` 并行读取 top-k 页面，减少总延迟
- **API 速率限制**：用 `aiolimiter` 或 Redis token bucket 限制每秒 LLM/VLM 请求数
- **用户级隔离**：Gradio 开 `concurrency_limit=5`，超出排队的请求用队列管理

### 7. 数据持久化（高）

**现状**：对话历史纯内存（`gradio_app.py:863` 的 `gr.State`），浏览器刷新/服务器重启全部消失。

**建议**：
- **轻量方案**：用 SQLite 持久化对话（`conversations` 表：id、title、created_at；`messages` 表：conv_id、role、content、citations、timestamp）
- **生产方案**：PostgreSQL + Redis。Redis 做热对话缓存，PG 做冷存储
- 支持对话导出（JSON/Markdown）
- 对话列表支持分页（目前全部加载到侧边栏）

### 8. 安全性（高）

**现状**：Gradio 无认证，任何人可访问并发起 LLM 调用（产生费用）。API 密钥明文落盘。

**建议**：
- **认证**：Gradio `launch(auth=(username, password))` 或集成 OAuth2（Google/GitHub）
- **密钥管理**：生产环境用 KMS 或环境变量注入，`.env` 仅限本地开发
- **输入校验**：限制 query 长度（max 2000 字符），过滤明显的 prompt injection 关键词
- **输出净化**：合成阶段增加一层 guardrail，检测并过滤幻觉引用（引用的 page_num 是否真实存在于 doc_memory）
- **HTTPS**：Gradio 前挂 Nginx/Caddy 做 TLS 终止

### 9. CI/CD 与部署（严重）

**现状**：无 Dockerfile，无 docker-compose，无 GitHub Actions，无任何自动化。

**建议**：
- **Dockerfile**：基于 `nvidia/cuda:12.1-runtime-ubuntu22.04`，安装 Python 3.10 + 依赖 + 模型权重挂载
- **docker-compose.yml**：定义 `app` + `redis` + `postgres` 服务
- **GitHub Actions CI**：
  ```yaml
  - lint: ruff check .
  - type: mypy agent/ tools/
  - test: pytest --cov
  - build: docker build && push
  ```
- **部署**：编写 `Makefile` 封装常用命令（`make dev`、`make test`、`make build`、`make deploy`）

### 10. 性能优化（中）

**现状**：ColPali 检索在每个文档上的 MaxSim 是串行的；索引构建批处理大小固定为 1。

**建议**：
- **检索并行化**：`tools/colpali_tool.py` 中将多文档的 embedding 拼接为单次 MaxSim，而非 for 循环逐个算
- **VLM 批量读取**：用 ThreadPoolExecutor 并行请求（top_k=5 从串行 5×2s=10s → 并行 2s）
- **索引构建**：`config.yaml` 中 `encode_batch_size: 1` 在有显存时调大到 4 或 8
- **模型量化**：ColQwen2 可考虑 `torch.float16` 或 `bitsandbytes` int8 量化，降低显存和推理延迟

### 11. 依赖管理（中）

**现状**：`requirements.txt` 使用 `>=` 范围约束，无锁文件。`fastapi` 和 `uvicorn` 被声明但从未使用。

**建议**：
- 迁移到 `pyproject.toml`：声明项目元数据、依赖分组（`[project.optional-dependencies]` 分离 dev/test/prod）
- 引入 `uv.lock` 或 `poetry.lock` 锁定依赖版本
- 清理虚假依赖：移除 `fastapi`、`uvicorn`（除非计划后续使用）
- 若后续引入 FastAPI，将 Gradio 挂载为 FastAPI 子应用，同时获得 OpenAPI 文档

### 12. 文档（低）

**现状**：README 全面，但缺乏 API 级文档和代码内 docstring。

**建议**：
- 为核心公开函数补 docstring（`executor_node`、`verifier_node`、`_maxsim`、`_encode_query`）
- 编写 `ARCHITECTURE.md` 独立描述架构决策（为什么用 ColPali 而非 text embedding、为什么 LangGraph 而非自定义状态机）
- `pyproject.toml` 中配置 `[tool.ruff]` 启用 docstring 检查规则

---

### 改进优先级路线图

| 阶段 | 内容 | 预估工期 |
|------|------|---------|
| **P0** 可上线 | 测试（单元+集成）、Docker 容器化、CI 流水线、修复多用户竞态条件 | 2-3 周 |
| **P1** 可靠性 | 重试机制+超时、LLM/VLM 缓存、结构化日志+LangSmith 追踪、Gradio 认证 | 2 周 |
| **P2** 性能 | VLM 并行读取、检索并行化、对话 SQLite 持久化 | 1 周 |
| **P3** 完善 | 指标监控+告警、KMS 密钥管理、速率限制、pyproject.toml 迁移 | 1-2 周 |

---

## 项目定位反思

### 亮点（不幼稚的地方）

- **视觉检索绕过 OCR**：这个设计决策本身不 trivial，传统 RAG 项目很少考虑。大多数 RAG 项目仍然 PDF → OCR → Text Chunk → Embedding，ColPali/ColQwen 方案是 2024 年才兴起的方向
- **Planner-Verifier-Synthesizer 编排**：不是简单的 QA 链，有真正的 agent 控制流和反思（reflexion）循环，属于 agentic RAG 范畴
- **MaxSim 后期交互**：比直接用 CLIP/ViT 做单向量检索强一个档次，理解了多向量检索的核心优势
- **金融领域的针对性**：财务指标计算（calculator 节点）、跨页一致性校验、引用溯源，这些领域适配不是通用 RAG 项目会考虑的

### 问题（看起来像玩具的地方）

- 零测试、无缓存、无部署方案 → 面试官一看就知道没上线过
- 对话历史纯内存、无认证、无监控 → 离"被人真正用过"差距明显
- 整体像个人学习项目而非可交付产品

---

## 前端选型与 Chainlit 整合规划

> **历史背景**：曾考虑 fork **kotaemon** 或 **Verba** 作为前端基底——前者不原生支持 ColPali/ColQwen 视觉检索且没有反思机制，后者无 agent 层、流水线固定。两者迁移代价都大、且需要在他们的抽象内重写后端。最终决策落在 **Chainlit**：Python 原生、对 LangGraph 一等公民、UI 层独立替换，**后端零改动**。
>
> 这一节记录 Chainlit 的能力盘点、与 findoc-agent 的对接方案、分阶段迁移规划。

---

### 1. Chainlit 是什么

Chainlit 是一个开源 Python 库，用 ~10 行代码就能给 LangChain / LangGraph / OpenAI / 自定义 agent 包一个生产级聊天 UI。和 Gradio/Streamlit 一样是声明式 Python + 自动热更新，但**专门为 agent / chain / RAG 场景设计**：

| 能力 | Chainlit 提供方式 |
|------|-------------------|
| **核心抽象** | `cl.Message`（用户/助手气泡）+ `cl.Step`（可折叠中间步骤），而非通用组件 |
| **多模态展示** | `cl.Image` / `cl.Pdf` / `cl.Text` / `cl.File` 元素直接挂在消息上——金融文档"页面图引用"原生支持 |
| **数据层** | SQLAlchemy / SQLite / PostgreSQL / Literal AI，会话历史 / 用户 / 反馈天然持久化 |
| **鉴权** | 密码、OAuth (Google/GitHub/Azure)、Header SSO 都是配置项 |
| **部署** | 官方 Docker 镜像 + Hugging Face Spaces 一键模板 |
| **可观测性** | 内置 Prompt Playground、step 时序图，外接 Literal AI 拿到完整 trace |

---

### 2. 为什么是 Chainlit（vs. 现有 Gradio / 其他候选）

| 维度 | 现有 Gradio | Chainlit | LangChain Agent Chat UI | CopilotKit |
|---|---|---|---|---|
| 语言 / 栈 | Python | Python | Next.js / TS | React / TS |
| LangGraph 适配 | 手写 `_stream_agent` | `LangchainCallbackHandler` 原生 | LangGraph SDK 直连 | CoAgents adapter |
| Agent 步骤可视化 | 手写 Markdown 拼字符串 | `cl.Step(name=...)` 自动折叠 | 内置 | 内置 |
| 页面图引用 | gallery 自己拼 | `cl.Image(path=..., display="inline")` | 自己写 React | 自己写 React |
| 多会话持久化 | ❌（`gr.State` 内存态） | ✅ Data Layer | 依赖后端 | 依赖后端 |
| 鉴权 | ❌ | ✅ 配置即用 | 自己接 | 自己接 |
| 主题 / 品牌化 | ✅ 高度自由 | ⚠️ 中等（config.toml + custom CSS） | 改 React | 改 React |
| 学习曲线 | 已会 | 1-2 天 | 1 周（含 LangGraph Server） | 1-2 周 |

**Chainlit 的取舍**：

✅ 把 DEVLOG 第 94-96 行三个遗留（会话持久化 / 多轮上下文 / 演示项目感）一次性补上
✅ 工业级差距分析里的 #3 可观测性 / #7 数据持久化 / #8 安全性（认证）前端侧自动达成
⚠️ amber 主题需要重做：`config.toml` + `public/stylesheet.css` 能改主色但灵活度不如 Gradio 写 CSS
⚠️ 三栏布局（左历史 / 中聊天 / 右多 Tab）不是 Chainlit 默认——侧边栏只有左侧；右侧"执行轨迹/引用/文档库"需要折中：执行轨迹用 `cl.Step`，引用用 inline `cl.Image`，文档库用 `cl.ChatProfile`

---

### 3. 与 LangGraph 的原生契合点

Chainlit 通过 `chainlit.langchain.callbacks.LangchainCallbackHandler` 拦截 LangChain 系（含 LangGraph）的所有 `on_chain_start` / `on_llm_start` / `on_tool_start` 等事件，自动转成 UI 元素：

```python
# app/chainlit_app.py（最小骨架）
import chainlit as cl
from agent.graph import compile_graph

graph = compile_graph()

@cl.on_chat_start
async def start():
    cl.user_session.set("graph", graph)

@cl.on_message
async def main(msg: cl.Message):
    g = cl.user_session.get("graph")
    cb = cl.LangchainCallbackHandler(
        stream_final_answer=True,
        to_ignore=["ChannelRead", "RunnableLambda"],
    )
    init_state = {
        "query": msg.content, "plan_cursor": 0, "reflexion_iter": 0,
        "is_sufficient": False, "retrieved_pages": [],
        "extracted_facts": [], "computed_values": [],
    }
    final = await g.ainvoke(init_state, config={"callbacks": [cb]})

    # 召回页作为 inline 图片元素挂在最终回答上
    elements = [
        cl.Image(
            path=hit["image_path"],
            name=f"{hit['doc_id']} p.{hit['page_num']}",
            display="inline",
        )
        for hit in final.get("retrieved_pages", [])
        if hit.get("image_path")
    ]
    await cl.Message(content=final["answer"], elements=elements).send()
```

**关键映射**：

- LangGraph 每个节点 enter/exit → 自动建一个 `cl.Step`，名字 = 节点名（`planner` / `executor` / `verifier` / `synthesizer`）
- 节点返回的 state delta → Step 的 `output` 字段（用户点开折叠区即可看）
- 节点内的 LLM / tool 调用 → 嵌套在所属 Step 下面的子 Step
- Reflexion 循环 → 多次 `executor` Step 自然按时间顺序排列

这正是当前 Gradio 版本里手写 `_summarize_node()` + `_format_tool_log()` 想做但做得不那么自然的事。

---

### 4. 与 findoc-agent 现有架构的映射表

| Gradio 现有实现 | Chainlit 等价物 | 备注 |
|---|---|---|
| `_app = compile_graph()` 模块级单例 | `cl.user_session.set("graph", g)` | 顺手解决工业级差距 #6 多用户竞态 |
| `_stream_agent` yield `(status, state, node, delta)` | `LangchainCallbackHandler` + `await graph.ainvoke()` | 节点级流式自动渲染为 Step |
| `chat_fn` 10 个 outputs | `@cl.on_message` async 函数 | 单入口，无 outputs 排序问题 |
| `gr.Chatbot` + 自写气泡 CSS | `cl.Message` | 自带 markdown / 复制 / 反馈按钮 |
| `gr.Gallery` "引用页" tab | `cl.Image(display="inline" or "side")` 列表挂在消息上 | 点击放大 |
| `_render_trace` Markdown | 自动从 `cl.Step` 生成 | 不需要手写 |
| `_format_tool_log` Markdown | 同上，按 Step 时间戳排列 | 不需要手写 |
| `gr.State(conversations)` 多会话 | Chainlit Threads + Data Layer | SQLite 落盘 |
| `_doc_summary_md` "文档库" tab | `cl.ChatProfile` 列表 | 每份年报作为一个 profile，进入即注入 `doc_filter` |
| 暂无认证 | `@cl.password_auth_callback` | 演示 demo 用账密足够 |
| amber 主题 | `.chainlit/config.toml` + `public/stylesheet.css` | 主色 `#d97706` 移植 |

---

### 5. 分阶段迁移规划

写到 DEVLOG 上对应：把 P5 标为 ✅(v1 Gradio) + 🚧(v2 Chainlit)，新增 P5.x 子阶段：

| 子阶段 | 目标 | 验收标准 | 预估 |
|---|---|---|---|
| **P5.1** 最小骨架 | 新建 `app/chainlit_app.py`，接通 `compile_graph()` | `chainlit run app/chainlit_app.py` 能问"贵州茅台 2023 年毛利率是多少"并看到 4 个节点的 Step 展开 | 0.5 天 |
| **P5.2** 多模态引用 | `cl.Image` 渲染 `retrieved_pages` | 最终回答下方挂召回页缩略图，点击放大 | 0.5 天 |
| **P5.3** 数据层 | 接 SQLite Data Layer，会话持久化 | 刷新浏览器后历史对话仍在；左侧栏列出所有 thread | 0.5 天 |
| **P5.4** 文档库 | `cl.ChatProfile` 把 14 份年报做成 profiles，选中后注入 `doc_filter` | 进入"贵州茅台 2023" profile，提问会自动过滤到该 doc_id | 0.5 天 |
| **P5.5** 品牌化 | `config.toml` + `public/stylesheet.css` 把主色调成 amber-600 | 视觉一致性：主按钮 / 用户气泡 / 高亮 | 0.5 天 |
| **P5.6** 鉴权 + 部署 | 密码鉴权 + Dockerfile（替换原计划的 Gradio 镜像） | `docker run findoc-chainlit` 在 8000 端口跑起来，登录后正常用 | 0.5 天 |

总计 ~3 天，比从零搭 kotaemon 集成省 ~60% 工时。

---

### 6. 改造影响面

只改两个目录：

- `app/`：新增 `chainlit_app.py`、`.chainlit/config.toml`、`public/stylesheet.css`；保留 `gradio_app.py` 一段时间作为 fallback，迁移稳定后再删
- `requirements.txt`：新增 `chainlit>=2.0`

**完全不改的部分**（这是 Chainlit 方案的核心价值）：

- `agent/`：图、节点、state、prompts、llm 工厂全部不动
- `tools/`：colpali_tool / vlm_tool / calculator 不动
- `ingestion/`：索引构建脚本不动
- `eval/`：评测代码不动
- `config.yaml`：不动

---

### 7. 风险与待验证项

| 项 | 风险 | 缓解 |
|---|---|---|
| Chainlit 版本 API 稳定性 | Gradio 6.14 踩过 API 变化坑（DEVLOG），Chainlit 2.x 是大版本 | `requirements.txt` 锁版本；P5.1 前先跑官方 langgraph 模板验证 |
| `stream_mode="updates"` vs Callback | 现有 `_stream_agent` 走 `stream_mode=updates`，Callback 走 events 模式，节点级 delta 颗粒度可能不同 | P5.1 验证：如果自动 Step 不够细，回退到手写 `cl.Step` 包 stream 循环 |
| 三栏布局还原度 | Chainlit 默认只有左栏 thread 列表，右侧多 Tab 需要 Custom Elements | 接受简化为左 thread + 主聊天 + 消息内联引用图；"文档库"用 ChatProfile 替代 |
| amber 主题灵活度 | 主题深度定制不如 Gradio 自由 | 接受"差不多 amber 即可"；不满意再 custom CSS 兜底 |
| 多用户并发下 graph 单例 | `_app` 模块级单例 + Chainlit 异步多 session | `cl.user_session` per-session 持有 graph；或确认 graph 无可变状态后共享 |
| 现有 ColQwen 模型加载耗时 | `colpali_tool.py` 模块级 lazy load，首请求慢 | 在 `@cl.on_chat_start` 触发预热，并用 `cl.Message` 提示加载中 |


---

## P7/P8 实施：ColQwen2 服务化 + Qdrant 多向量存储

> 实施日期：2026-05-03。本节是 P7/P8 的详细代码逻辑说明，配合 DEVLOG §"2026-05-03 P7 / P8 / P9 实施记录" 使用。DEVLOG 讲改了什么文件、为什么这么改；LEARNLOG 讲代码怎么运作的。

---

### 一、架构全景图（改造前 vs 改造后）

**改造前**（P7/P8 之前）：
```
Chainlit worker (GPU)                  
  ├── colpali_tool.py                  
  │     ├── 加载 ColQwen2 模型（~8GB VRAM）
  │     ├── encode_query(text) → embedding [Tq, 128]
  │     ├── 加载 data/index/*/embeddings.pt（mmap → 内存）
  │     └── Python _maxsim(q_emb, doc_emb) → top-k pages
  └── agent 节点（Planner / Executor / Verifier / Synthesizer）
```

**改造后**（P7 + P8 全启用）：
```
Chainlit worker (CPU-only)             ColQwen Service (GPU, Litserve)
  ├── colpali_tool.py                  ├── POST /predict                     
  │     ├── httpx → /predict ──────────►│     ├── encode_query(text) → emb    
  │     └── httpx → Qdrant ──┐         │     └── encode_pages(images) → emb  
  └── agent 节点              │         └─────────────────────────────────────
                              │
                     Qdrant (Docker)
                       ├── collection: findoc_pages
                       ├── MultiVector + MaxSim (server-side)
                       └── doc_id / page_num / image_path payload
```

**默认行为不依赖任何新服务**：`config.retriever.backend: in_memory` + `config.services.colqwen_url: ""` → 完全等同于改造前。

---

### 二、P7 核心代码逻辑

#### 2.1 `ingestion/model_loader.py` — 共享模块

这是 P7 的基石。原本 `build_index.py` 和 `colpali_tool.py` 各有一份 `_load_model_and_processor()`，P7 提取到共享模块。

```
load_model_and_processor(lora_path=None)
  │
  ├── 读 config.retriever.backbone → 决定加载 ColQwen2 还是 ColPali
  ├── 读 config.retriever.dtype → torch.bfloat16 / float16 / float32
  ├── resolve_path(config.retriever.model_name) → 绝对路径
  ├── ModelCls.from_pretrained(model_name, torch_dtype=dtype, device_map=device)
  │     └── .eval()
  ├── 如有 lora_path（参数或 config）：PeftModel.from_pretrained → merge_and_unload
  └── ProcessorCls.from_pretrained(model_name)
  → (model, processor)

encode_query(model, processor, query)
  │
  ├── processor.process_queries([query]) → batch dict
  ├── batch.to(model.device)
  ├── with torch.no_grad(): emb = model(**batch)
  └── emb.to("cpu", dtype=torch.float16)[0]  → [Tq, 128]

encode_pages(model, processor, image_paths, batch_size)
  │
  ├── for batch in image_paths:
  │     ├── Image.open(p).convert("RGB") for p in batch
  │     ├── processor.process_images(images) → batch dict
  │     ├── model(**batch) → emb [B, T, 128]
  │     ├── emb.cpu().float16()
  │     ├── img.close(); torch.cuda.empty_cache()
  │     └── append to chunks[]
  ├── 对齐 token 数：所有 chunk 的 dim=1（token 数）可能不同
  │     └── max_tokens = max(c.shape[1])
  │     └── pad 较短的 chunk 到 max_tokens（zero-padding）
  └── torch.cat(chunks, dim=0) → [P, max_T, 128]
```

**为何 token 数会不同**：ColQwen2 的 vision encoder 对每张图片的输出 token 数取决于图片分辨率。不同页面可能渲染出不同尺寸的 PNG，导致 token 数差异。pad 到统一长度后才能 torch.cat。

#### 2.2 `services/colqwen_server.py` — Litserve GPU 服务

Litserve 是 Lightning AI 的模型服务框架，核心抽象是 `LitAPI` 四个方法：

```
class ColQwenLitAPI(ls.LitAPI):

    setup(device)
      ├── 调用 model_loader.load_model_and_processor()
      ├── self.model, self.processor 驻留显存
      └── 在 worker 启动时执行一次，非首请求时

    decode_request(request: dict)
      ├── request["action"] == "encode_query" → ("encode_query", request["query"])
      └── request["action"] == "encode_pages" → ("encode_pages", [Path(p) for p in request["image_paths"]])

    predict(inputs: tuple)
      ├── action == "encode_query" → encode_query(model, processor, data)
      │     └── 返回 {"embedding": emb.tolist()}  # serializable
      └── action == "encode_pages" → encode_pages(model, processor, data, batch_size=1)
            └── 返回 {"embeddings": embs.tolist()}  # serializable

    encode_response(output) → output  # 原样返回 dict
```

**Litserve 配置要点**：
- `accelerator="auto"` — 自动检测 CUDA
- `devices=1, workers_per_device=1` — 单 GPU 单 worker，显存只够一份模型
- `timeout=False` — 页面编码可能耗时数分钟，不限超时
- 自动生成 `GET /health` — 在 `setup()` 完成前返回 unhealthy

**encode_pages 为何 batch_size=1**：页面图片的 token 数不一致，batch > 1 时 processor 会 pad 到统一尺寸，但 pad 区域生成的 token 是垃圾信息。batch_size=1 避免浪费计算。

#### 2.3 `tools/colpali_tool.py` — 远程编码集成

核心改动在 `_encode_query()` 的决策逻辑：

```
_encode_query(query)
  │
  ├── emb = _encode_query_remote(query)
  │     ├── 读 config.services.colqwen_url
  │     ├── 若为空 → return None（不尝试远程）
  │     ├── httpx.post(f"{url}/predict", json={"action":"encode_query","query":query}, timeout=30)
  │     ├── resp.json()["embedding"] → torch.tensor(..., dtype=float16)
  │     └── 任何异常（连接拒绝/超时/HTTP 错误）→ log warning + return None
  │
  ├── if emb is not None: return emb  # 远程成功
  │
  └── _ensure_model_loaded()          # 远程失败或未配置 → fallback 本地
        └── _local_encode_query(_state["model"], _state["processor"], query)
```

**`_ensure_model_loaded()` vs `_ensure_loaded()` 的拆分**：

```
之前的 _ensure_loaded():
  加载索引 + 加载模型  ← 耦合在一起

P7 后的 _ensure_loaded():
  只加载索引（数据依赖）
  → 被 _in_memory_retrieve() 调用

P7 后的 _ensure_model_loaded():
  只加载模型（仅当需要本地编码时）
  → 被 _encode_query() 的 fallback 路径调用
```

拆分原因：远程编码时不需要本地模型，只需索引数据（.pt 文件在内存做 MaxSim）。索引数据轻量（mmap），模型是重量的（~8GB VRAM）。

#### 2.4 `ingestion/build_index.py` — 远程编码选项

```
main()
  ├── args.colqwen_url ← CONFIG["services"]["colqwen_url"]（默认为 ""）
  ├── if colqwen_url:
  │     model, processor = None, None    # 不加载本地模型
  │     logger.info(f"Using remote ColQwen Service at {colqwen_url}")
  │   else:
  │     model, processor = load_model_and_processor(args.lora_path)
  │
  └── for each doc:
        index_doc(model, processor, doc_dir, index_dir, batch_size, colqwen_url)

index_doc(...)
  ├── 幂等检查：embeddings.pt + meta.json 都存在 → skip
  ├── if colqwen_url:
  │     embeddings = _encode_pages_remote(image_paths, colqwen_url, batch_size)
  │   else:
  │     embeddings = encode_pages(model, processor, image_paths, batch_size)
  └── torch.save(...) + meta.json.write_text(...)

_encode_pages_remote(image_paths, colqwen_url, batch_size)
  ├── for batch in image_paths:
  │     httpx.post(f"{url}/predict",
  │       json={"action":"encode_pages", "image_paths": [str(p) for p in batch]},
  │       timeout=600)  # 长超时：大批量页面编码
  │     → data["embeddings"] → torch.tensor(float16)
  └── pad + torch.cat（同本地编码的 token 对齐逻辑）
```

**注意**：image_paths 传的是文件系统路径字符串，不是 base64。这是因为在部署环境下，ingestion 脚本和 ColQwen Service 共享文件系统（同一台机器或共享卷）。

---

### 三、P8 核心代码逻辑

#### 3.1 `ingestion/push_to_qdrant.py` — 索引迁移

```
main()
  ├── client = QdrantClient(url=config.retriever.qdrant.url)
  ├── get_or_create_collection(client, collection, recreate)
  │     ├── if recreate: client.delete_collection(collection)
  │     ├── if collection exists: return
  │     └── client.create_collection(
  │           vectors_config=VectorParams(size=128, distance=Distance.DOT),
  │           multivector_config=MultiVectorConfig(
  │             comparator=MultiVectorComparator.MAX_SIM))
  │
  └── for doc_dir in data/index/*/:
        push_doc(client, collection, doc_dir)

push_doc(client, collection, doc_dir)
  ├── torch.load("embeddings.pt", weights_only=True)
  │     → embeddings: [P, T, 128], page_nums: [P]
  ├── json.load("meta.json") → doc_id, pages[]
  │
  ├── for i in range(P):
  │     point = PointStruct(
  │       id = abs(hash(f"{doc_id}__p{page_num}")) & 0x7FFFFFFFFFFFFFFF,
  │       vector = embeddings[i].tolist(),  # [T, 128] → list[list[float]]
  │       payload = {doc_id, page_num, image_path})
  │     points.append(point)
  │
  └── client.upsert(collection, points, wait=True)
```

**关键设计决策**：

1. **`Distance.DOT` 而非 `Distance.COSINE`**：Python 的 `_maxsim` 用 `torch.einsum("td,pkd->ptk", q, d)` 计算 dot product，不做归一化。Qdrant 的 DOT 距离等价于未归一化的 dot product，和 Python 实现一致。如果用 COSINE，Qdrant 会先归一化向量再算 dot product，结果不同。

2. **Point ID 用 hash**：`abs(hash(f"{doc_id}__p{page_num}")) & 0x7FFFFFFFFFFFFFFF` 保证：
   - 确定性（同样 doc_id + page_num 总是同样 ID）
   - 正整数（Qdrant 要求 int64 > 0）
   - 幂等性（再跑一次 push_to_qdrant 是 upsert 而非 insert，不会重复）

3. **`weights_only=True`**：torch.load 安全参数，防止 pickle 反序列化攻击。

#### 3.2 `tools/colpali_tool.py` — Qdrant 检索后端

```
_qdrant_retrieve(query, top_k, doc_filter)
  │
  ├── client = QdrantClient(url=config.retriever.qdrant.url)
  ├── q_emb = _encode_query(query)  # [Tq, 128]，可能走远程或本地
  ├── query_vector = q_emb.numpy().tolist()  # list[list[float]]
  │
  ├── 构造 filter（如果 doc_filter 不为空）：
  │     ├── 单个 doc_id → Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=...))])
  │     └── 多个 doc_id → Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=[...]))])
  │
  ├── results = client.query_points(
  │       collection_name=collection,
  │       query=query_vector,         # 多向量 query
  │       query_filter=filter,        # 可选 doc 过滤
  │       limit=top_k,                # 只取 top-k
  │       with_payload=True)          # 带上 doc_id / page_num / image_path
  │
  └── for scored in results.points:
        PageHit(doc_id=payload["doc_id"], page_num=payload["page_num"],
                score=scored.score, image_path=payload["image_path"])
```

**Qdrant 服务端发生了什么**：

1. 接收 query_vector `[Tq, 128]`（多向量 query）
2. 对 collection 中每个 point（一页），该 point 的 vector 也是 `[Td, 128]`
3. 计算 MaxSim：对每个 query token，找 page token 中最大 dot product；然后 sum over query tokens
4. 返回 score 最高的 top_k 个 point，连同 payload

这完全等价于 Python 的 `einsum("td,pkd->ptk") → max(dim=-1) → sum(dim=-1)`，但计算在 Qdrant 服务端完成，不占用 Python 进程的 CPU。

**fallback 链**（在 `colpali_retrieve()` 中）：

```python
if backend == "qdrant":
    try:
        return _qdrant_retrieve(query, top_k, doc_filter)
    except Exception:
        logger.error("Qdrant retrieval failed, falling back to in_memory")
        # 继续往下走 _in_memory_retrieve
return _in_memory_retrieve(query, top_k, doc_filter)
```

任何 Qdrant 异常（连接失败、collection 不存在、超时）都自动 fallback 到本地 in_memory 检索，不会让 agent 崩溃。

#### 3.3 `config.yaml` 新增配置

```yaml
retriever:
  backend: in_memory           # "in_memory" | "qdrant"
  qdrant:
    url: http://localhost:6333
    collection_name: findoc_pages

services:
  colqwen_url: ""              # "" = 本地模型, "http://localhost:8000" = 远程
```

**配置优先级**：
- `colqwen_url: ""` → 本地加载模型（和 P7 之前一致）
- `colqwen_url: "http://localhost:8000"` + `backend: in_memory` → 远程编码 + 本地 MaxSim
- `colqwen_url: "http://localhost:8000"` + `backend: qdrant` → 远程编码 + Qdrant MaxSim
- `colqwen_url: ""` + `backend: qdrant` → 本地编码 + Qdrant MaxSim

四种组合全部可用，用户根据需要选择。

---

### 四、全链路数据流（以一次查询为例）

用户输入 "贵州茅台2023年毛利率"：

```
1. Chainlit @cl.on_message
     │
     ├── _GRAPH.astream(state, stream_mode="updates")
     │
     ├── [planner] → {"plan": [SubTask(...)], "plan_cursor": 0}
     │     └── cl.Step(name="🧠 Planner · 产出1步计划: 贵州茅台2023年毛利率…")
     │           output = "**执行计划** — 1 步\n1. `贵州茅台2023年毛利率` [moutai_2023]"
     │
     ├── [executor] → {"retrieved_pages": [PageHit...], "extracted_facts": [...], ...}
     │     │
     │     │  colpali_retrieve("贵州茅台2023年毛利率", top_k=5)
     │     │    ├── backend=="qdrant"?
     │     │    │   ├── _encode_query_remote(query) → httpx → ColQwen Service → emb [Tq,128]
     │     │    │   └── _qdrant_retrieve → httpx → Qdrant query_points → top-5 pages
     │     │    └── backend=="in_memory"?
     │     │        ├── _encode_query → remote (if configured) or local model
     │     │        └── _in_memory_retrieve → Python MaxSim over .pt files
     │     │
     │     │  vlm_read_page(each retrieved page) → Qwen VLM → structured facts
     │     │
     │     └── cl.Step(name="🔧 Executor · 检索5页 · 抽取5事实 · 计算0")
     │           output = "**检索页面** (5)\n- moutai_2023 p.042 score=9.87\n..."
     │
     ├── [verifier] → {"is_sufficient": True, "reflexion_iter": 1, ...}
     │     └── cl.Step(name="🔍 Verifier · ✅ 充分")
     │           output = "**判断**: ✅ 信息充分\n\n轮次: 1"
     │
     ├── [synthesizer] → {"answer": "...毛利率为94.12%...", "citations": [...]}
     │     └── cl.Step(name="✍️ Synthesizer · 生成答案(52字符 · 5引用)")
     │           output = "**生成答案** — 52 字符 · 5 处引用\n\n---\n\n..."
     │
     └── cl.Message(content=answer + citations, elements=[cl.Image(...), ...])
```

---

### 五、操作命令速查

```bash
# ------ P7: ColQwen Service ------
# 启动（GPU 机器上）
python -m services.colqwen_server --port 8000
# 健康检查
curl http://localhost:8000/health
# 测试 query 编码
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"action":"encode_query","query":"贵州茅台2023年毛利率"}'

# ------ P8: Qdrant ------
# 启动
docker compose up -d qdrant
# 推送索引（幂等，可重复执行）
python -m ingestion.push_to_qdrant
# 重建 collection + 重新推送
python -m ingestion.push_to_qdrant --recreate
# 只推送特定文档
python -m ingestion.push_to_qdrant --only moutai_2023

# ------ 切回 in_memory ------
# 在 config.yaml 中设 backend: in_memory 即可
# 已有的 .pt 文件不受影响
```


---

## P10 前后端分离 + 实时进度推送

> 实施日期：2026-05-03。本节记录 FastAPI 后端 + Chainlit 前端分离的架构设计、SSE 流式协议、进度事件机制，以及 Executor 冷启动问题的排查与修复。

---

### 一、改造前后架构对比

**改造前**（P9 之后）：
```
chainlit run app/chainlit_app.py   ← 单进程
  ├── import agent.graph.compile_graph()
  ├── _GRAPH.astream()  同进程
  ├── tools.colpali_tool  同进程 (ColQwen2 加载到同一 GPU)
  └── 无法独立扩缩容 / 前端崩 = 全系统崩
```

**改造后**（P10）：
```
终端 1: uvicorn backend.server:app --port 8001
  ├── FastAPI + SSE
  ├── agent/ tools/ ingestion/ (零改动)
  ├── ColQwen2 模型启动时预加载
  └── POST /api/v1/query → SSE 流

终端 2: chainlit run app/chainlit_app.py -w
  ├── 零 agent import (仅 httpx + SSE 消费)
  ├── 所有业务逻辑通过 HTTP 调用
  └── 纯 UI 层：Step 渲染 / 消息展示 / 图片引用
```

**agent/ tools/ ingestion/ 三个目录一行未改**——后端只是把它们包了一层 HTTP。

---

### 二、SSE 流式协议设计

选择 SSE（Server-Sent Events）而非 WebSocket 或 REST polling：

| 考量 | SSE | WebSocket | REST polling |
|------|-----|-----------|-------------|
| 通信方向 | 后端→前端单向流 | 双向 | 前端→后端拉取 |
| Agent 输出特点 | 完美匹配（4 节点顺序输出） | 多余 | 非实时 |
| 实现复杂度 | FastAPI `StreamingResponse` 原生支持 | 需要 ws 库 | 简单但体验差 |
| 代理/网关兼容 | HTTP，天然兼容 | 需要 upgrade | 兼容 |
| 断线重连 | 浏览器原生支持 | 需手写 | 需手写 |

**SSE 事件格式**：

```
event: node
data: {"node":"planner","summary":"产出1步计划: ...","content":"**执行计划** — 1 步\n..."}

event: status
data: {"type":"status","message":"正在 MaxSim 检索 14 份文档..."}

event: status
data: {"type":"status","message":"VLM 正在读取 p009.png..."}

event: node
data: {"node":"executor","summary":"检索5页 · 抽取5事实 · 计算0","content":"..."}

event: node
data: {"node":"verifier","summary":"✅ 充分","content":"..."}

event: node
data: {"node":"synthesizer","summary":"生成答案(52字符 · 5引用)","content":"..."}

event: done
data: {"type":"done","answer":"...","citations":[...],"retrieved_pages":[...]}
```

四种 event 类型：
- `node`：一个 LangGraph 节点完成，携带 `summary`（Step 标题，35 字截断）+ `content`（Step 展开内容，markdown）
- `status`：进度更新，Executor 内部的长时间操作（模型加载、MaxSim、VLM 页面读取）的实时状态
- `done`：Agent 执行完成，携带最终答案、引用列表、检索页面列表
- `error`：执行异常，携带错误信息

---

### 三、后端实现：`backend/server.py`

#### 3.1 启动预加载

```python
@app.on_event("startup")
async def startup():
    colqwen_url = CONFIG.get("services", {}).get("colqwen_url", "")
    if colqwen_url:
        return  # P7 远程模式，跳过本地加载
    from tools.colpali_tool import preload
    await asyncio.to_thread(preload)  # 阻塞直到模型就绪
```

`asyncio.to_thread` 将同步的模型加载放到线程中，避免阻塞事件循环。Uvicorn 在 startup 事件完成前不 accept 连接，因此首个请求到达时模型一定已在内存。

#### 3.2 SSE 流与线程模型

核心矛盾：Agent 图执行是同步阻塞的（`_GRAPH.stream` 一次 `for` 迭代可能卡 10-70 秒），但 SSE 需要持续推送进度事件。

解决方案：**后台线程 + 双队列**

```
┌─ Thread: _run_agent_sync ─────────────────────┐
│                                                 │
│  _GRAPH.stream(state)                           │
│    ├─ planner 完成 → node_queue.put(...)        │
│    ├─ executor 执行中...                         │
│    │    └─ colpali_tool._report_progress() ──┐  │
│    │    └─ vlm_tool._report_progress() ─────┤  │
│    │                                         │  │
│    ├─ executor 完成 → node_queue.put(...)    │  │
│    └─ ... → node_queue.put(("__done__",..))  │  │
│                                                 │
└─────────────────────────────────────────────────┘
                     │                │
              node_queue         progress_queue
            (node results)    (进度消息字符串)
                     │                │
┌─ Async event loop (主线程) ────────────────────┐
│                                                 │
│  while True:                                    │
│    drain progress_queue  → yield event: status  │
│    drain node_queue       → yield event: node   │
│    if __done__: break                          │
│    await asyncio.sleep(0.3)  ← 300ms 轮询间隔  │
│                                                 │
└─────────────────────────────────────────────────┘
```

关键代码（`backend/server.py`）：

```python
node_queue: queue.Queue = queue.Queue()
progress_queue: queue.Queue = queue.Queue()

def _on_progress(msg: str) -> None:
    progress_queue.put(msg)

def _run_agent_sync(state: dict) -> None:
    # 接线进度回调
    set_colpali_hook(_on_progress)
    set_vlm_hook(_on_progress)

    accumulated = dict(state)
    for chunk in _GRAPH.stream(state, stream_mode="updates"):
        for node_name, delta in chunk.items():
            node_queue.put((node_name, delta))
            # merge delta to accumulated...

    node_queue.put(("__done__", accumulated))

async def event_stream():
    thread = threading.Thread(target=_run_agent_sync, args=(state,), daemon=True)
    thread.start()

    while True:
        # 优先排空进度队列
        while True:
            try:
                msg = progress_queue.get_nowait()
                yield f"event: status\ndata: {json.dumps({'type':'status','message':msg})}\n\n"
            except queue.Empty:
                break

        # 检查节点结果
        try:
            node_name, data = node_queue.get_nowait()
            if node_name == "__done__":
                break
            # yield event: node ...
        except queue.Empty:
            yield ": keepalive\n\n"   # 防止代理超时断开
            await asyncio.sleep(0.3)  # 轮询间隔
```

#### 3.3 其他端点

```python
@app.get("/api/v1/docs")
async def list_docs():
    # 读 doc_memory.json → 返回文档列表
    # 前端 @cl.on_chat_start 调用此端点展示已索引文档

@app.get("/api/v1/health")
async def health():
    # 返回 {"status":"ok", "docs_count":14, "backend":"in_memory"}
```

---

### 四、前端实现：`app/chainlit_app.py`

#### 4.1 纯前端原则

链式导入检查——前端只允许 import：
- `chainlit`（UI 框架）
- `httpx`（HTTP 客户端）
- `agent.config` → `CONFIG`（读后端 URL）+ `PAGES_DIR`（解析图片路径）
- 标准库（`json` / `pathlib` / `sys`）

**不允许** import `agent.graph` / `tools.*` / `ingestion.*`。

#### 4.2 SSE 消费

```python
@cl.on_message
async def on_message(msg: cl.Message):
    _status_step = None

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", f"{_BACKEND_URL}/api/v1/query",
                                  json={"query": query}, timeout=300.0) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = json.loads(line[5:].strip())
                etype = data.get("type", "")

                if etype == "status":
                    # 动态更新进度 Step（名字随消息变化）
                    msg_text = data["message"]
                    if _status_step is None:
                        _status_step = cl.Step(name=f"⏳ {msg_text}", type="tool")
                        await _status_step.send()
                    else:
                        _status_step.name = f"⏳ {msg_text}"
                        await _status_step.update()

                elif etype == "done":
                    if _status_step:
                        _status_step.name = "✅ 完成"
                        await _status_step.update()
                    # 提取最终答案、引用、检索页面
                    # → cl.Message(content=answer, elements=images)

                elif "node" in data:
                    # 创建可折叠 Step，名字 = emoji + 35字摘要
                    label = _NODE_LABELS[data["node"]]
                    step_name = f"{label} · {_truncate(data['summary'], 35)}"
                    async with cl.Step(name=step_name, type="tool") as step:
                        step.output = data["content"]
```

#### 4.3 进度 Step 的 UX

进度 Step 是一个名字随消息变化的 `cl.Step`：

```
⏳ 正在 MaxSim 检索 14 份文档...        ← 名字实时更新
⏳ VLM 正在读取 p009.png...
⏳ VLM 完成 p009.png
⏳ VLM 正在读取 p016.png...
...
✅ 完成                                  ← done 事件时切换
```

用户不再面对 80 秒空白——每一步都看得到。

---

### 五、进度回调机制

**问题**：`astream` 在 Executor 被模型加载（~70s）和 VLM API 调用（~10s）阻塞，主循环无法推送进度。而工具函数是同步的，不能 await。

**方案**：模块级回调 + 线程安全队列，最小化侵入。

#### 5.1 工具侧（`tools/colpali_tool.py`、`tools/vlm_tool.py`）

```python
# 模块级全局，默认 None（无进度推送）
_progress_hook: "callable | None" = None

def set_progress_hook(hook):
    global _progress_hook
    _progress_hook = hook

def _report_progress(msg: str):
    if _progress_hook:
        _progress_hook(msg)   # 同步回调，约 1μs

# 在长操作点调用 _report_progress
def _ensure_model_loaded():
    _report_progress("正在加载 ColQwen2 模型到 GPU（约 60 秒）...")
    model, processor = _load_model()
    _report_progress("ColQwen2 模型就绪")

def vlm_read_page(image_path, instruction):
    fname = Path(image_path).name
    _report_progress(f"VLM 正在读取 {fname}...")
    result = _call_openai_compat(image_path, instruction, api_key)
    _report_progress(f"VLM 完成 {fname}")
    return result
```

**侵入性**：每个文件加 ~15 行，只加在确实耗时的操作点。不影响任何现有函数签名和返回值。

#### 5.2 后端侧（`backend/server.py`）

```python
def _on_progress(msg: str) -> None:
    progress_queue.put(msg)  # thread-safe

def _run_agent_sync(state):
    set_colpali_hook(_on_progress)
    set_vlm_hook(_on_progress)
    # ... run graph ...
```

回调在后台线程中执行，`_on_progress` 只是往 `queue.Queue` push 一个字符串（thread-safe, lock-free），开销 < 1μs。

---

### 六、关键验证数据

首次请求（模型已在 startup 预加载）：

```
[  5.9s] 🧠 Planner        gap 5.9s  ← DeepSeek API
        ⏳ 正在 MaxSim 检索 14 份文档...    ← 实时推送！
        ⏳ VLM 正在读取 p001.png...
        ⏳ VLM 完成 p001.png
        ... (×5 pages)
[ 16.0s] 🔧 Executor       gap 10.1s ← MaxSim + VLM × 5
[ 20.8s] 🔍 Verifier       gap  4.8s ← DeepSeek API
[ 25.4s] ✍️ Synthesizer    gap  4.6s ← DeepSeek API
[total: ~25s]  冷启动需额外 +70s（模型加载），但 startup 已预加载
```

验证 SSE 进度事件数量（curl 实测）：

```
event: node        ← planner
event: status      ← 11 条进度消息，在 Executor 执行期间实时推送
event: status      ←   包括 MaxSim 进度 + 10 条 VLM 页面读取进度
...
event: status
event: node        ← executor
event: node        ← verifier
event: node        ← synthesizer
event: done
```

---

### 七、操作命令速查

```bash
# ------ 后端 ------
PYTHONPATH=. uvicorn backend.server:app --host 0.0.0.0 --port 8001
# 启动日志:
#   Preloading ColQwen2 model and indexes ...
#   ColQwen2 model warm — ready to serve
#   Application startup complete.

# ------ 前端 ------
chainlit run app/chainlit_app.py -w
# 需后端先启动，否则显示 "无法连接后端" 提示

# ------ 测试 SSE（不用前端）------
curl -X POST http://localhost:8001/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"贵州茅台2023年毛利率"}' \
  --no-buffer

# 期望输出：event: node ... event: status ... event: node ... event: done
```