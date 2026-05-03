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

## 九、亮点（值得保留的设计）

- **视觉检索绕过 OCR**：ColPali/ColQwen 是 2024 年才兴起的方向，多数 RAG 项目仍走 OCR + 文本分块
- **Planner-Verifier-Synthesizer 显式状态机**：不是简单 QA 链，有真正的 reflexion 循环
- **MaxSim 后期交互**：比 CLIP/ViT 单向量检索强一档
- **金融领域适配**：calculator AST 求值、跨页一致性校验、引用溯源
- **三层服务边界清晰**：ColQwen Service（GPU）/ Qdrant（向量库）/ FastAPI（业务）/ Chainlit（UI），任何一层可独立替换
