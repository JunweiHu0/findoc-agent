# FinDoc Agent

> 面向视觉密集型金融文档（年报、招股书、研报）的多模态检索增强 Agent。以自微调的 **ColPali** 作为核心视觉检索工具，基于 **LangGraph** 构建 Planner–Executor–Verifier 三段式状态机，支持跨文档对比、表格/图表解读与带反思机制的多轮检索。

[![Demo](https://img.shields.io/badge/demo-video-blue)]() [![Python](https://img.shields.io/badge/python-3.10+-green)]() [![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## 1. 项目背景与动机

金融年报这类文档有两个特点让传统文本 RAG 表现很差：

1. **版式复杂、信息密集**：财务报表、趋势图、组织架构图、合并报表附注，文本 chunk 后语义被切碎，OCR 还原结构损耗大。
2. **跨页跨文档推理需求强**：用户问"对比 A、B 两家公司近三年毛利率趋势"，本质是一个需要多步检索 + 数值计算 + 对比生成的任务，单轮 RAG 无法覆盖。

本项目的核心思路：**用 ColPali 解决"看得见"的问题，用 Agent 解决"想得清"的问题**。

- ColPali 直接以页面图像为检索单位，绕过 OCR/版面解析，对表格、图表、版式复杂页面召回显著优于文本检索；
- Agent 层负责任务规划、工具调用、反思与重检索，把"召回的页面"转化为"结构化的回答"。

## 2. 系统架构

```
                ┌─────────────────────────┐
                │   User Query (NL)       │
                └────────────┬────────────┘
                             ▼
                  ┌──────────────────┐
                  │     Planner      │  拆解为有序子任务
                  └────────┬─────────┘
                           ▼
              ┌────────────────────────┐
   ┌────────► │       Executor         │  调用工具执行子任务
   │          └────────┬───────────────┘
   │                   ▼
   │         ┌────────────────────┐
   │         │  Tool Layer        │
   │         │  ├ colpali_retrieve│
   │         │  ├ vlm_read_page   │
   │         │  └ calculator      │
   │         └────────┬───────────┘
   │                  ▼
   │         ┌────────────────────┐
   │  不充分  │     Verifier       │  反思：召回是否足够？
   └─────────┤  (Reflexion Node)  │  数值是否一致？
             └────────┬───────────┘
                      ▼ 充分
             ┌────────────────────┐
             │    Synthesizer     │  生成最终答案 + 引用
             └────────┬───────────┘
                      ▼
                  Final Answer
```

整套流程用 **LangGraph** 实现为一个显式状态机，每个节点的输入/输出/状态变更都可观测、可回放。

## 3. 关键设计

### 3.1 Planning：轻量级 Decomposition Planner

不采用 ReAct 式的"边想边做"，而是在入口先做一次显式任务分解：

- **输入**：用户原问题 + 文档元信息（公司、年份、可用文件列表）
- **输出**：一个有序的子任务列表，每个子任务带 `target_doc`、`sub_query`、`expected_output_schema`
- **为什么这么做**：金融问答的子任务边界是清晰的（"取数 → 计算 → 对比"），先规划再执行比 ReAct 在该场景下更稳定，token 开销也更低。
- **降级策略**：若 Planner 输出无法解析为合法 JSON，自动回退为单步执行（把原问题直接塞进 Executor），保证鲁棒性。

### 3.2 Tool Use：三个聚焦的工具，而非"工具大杂烩"

| Tool | 职责 | 实现 |
|---|---|---|
| `colpali_retrieve(query, top_k, doc_filter)` | 视觉检索，返回 top-k 页面（图像 + 元信息） | 自微调 ColPali + late interaction（MaxSim）|
| `vlm_read_page(page_image, instruction)` | 对召回页做结构化抽取（数字、趋势、表头） | Qwen2.5-VL-7B / GPT-4o（可切换）|
| `calculator(expr)` | 安全数值计算（毛利率、同比、环比） | 基于 AST 的受限 eval，禁用任意函数调用 |

设计原则：**Tool 数量保持在 ≤ 3，每个 Tool 边界正交**。多 Tool 在 demo 阶段看起来酷，但会显著增加 Planner 选错工具的概率，且面试时讲不清边界。

### 3.3 Memory：分层短/长期记忆

- **Working Memory（短期）**：LangGraph 的 `State` 对象贯穿整次对话，存储已检索页面、已抽取数据、子任务执行轨迹。供 Verifier 判断"是否已掌握足够信息"。
- **Episodic Memory（中期）**：单次 session 内的 QA 历史以摘要形式压缩后挂载在 State 上，支持追问场景（"那它的研发投入呢？"）。
- **Document Memory（长期）**：每份 PDF 在 ingest 阶段生成一份元信息摘要（公司名、年份、关键章节页码索引），缓存到本地 JSON。Planner 调用时读取，避免每次都让 LLM 去"猜"文档结构。

刻意没有引入向量数据库做长期 memory：当前数据量级（数十份年报、数千页）下，本地 `.pt` + 内存 MaxSim 即可在 100ms 内完成检索，引入 Qdrant/Milvus 是过度工程化。

### 3.4 Reflexion：基于充分性 + 一致性的双重校验

Verifier 节点是本项目的核心差异点。每轮 Executor 执行完后，Verifier 检查两件事：

1. **充分性（Sufficiency）**：当前 working memory 里的证据是否足以回答原问题？
   - 用 LLM 做一次结构化判断，输出 `{is_sufficient: bool, missing_info: str}`
   - 不充分时，把 `missing_info` 作为新的 sub_query 回传给 Executor
2. **一致性（Consistency）**：多页召回到的数字是否一致？跨文档对比时的口径是否一致（合并报表 vs 母公司报表）？
   - 不一致时，触发"溯源"——让 Executor 重新读取原页定位差异。

**防死循环机制**：`max_reflexion_iterations = 3`，超过后强制进入 Synthesizer 并在最终答案中标注"低置信度"。

这个设计参考了 Reflexion (Shinn et al., 2023) 的思想，但简化为面向"检索完备性"的领域定制版本，而非通用的 self-critique。

### 3.5 引用与可追溯性

最终回答中每一个事实性陈述都会附带 `[doc_id, page_num]` 引用，前端 demo 中点击引用直接跳转到该页截图。这一点在金融场景是合规级别的硬需求，也是把本项目和"玩具 demo"区别开的工程细节。

## 4. 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| Agent 编排 | **LangGraph** | 状态机显式，节点可观测，便于实现 reflexion 循环；优于 LangChain `AgentExecutor` 的黑盒和 AutoGen 的多 agent 对话范式 |
| 视觉检索 | **ColPali (自微调)** | 已在金融年报上做 LoRA 微调，page-level multi-vector + late interaction |
| VLM | Qwen2.5-VL-7B / GPT-4o | 前者本地部署可控，后者作为 fallback |
| LLM (Planner/Verifier) | GPT-4o-mini / DeepSeek-V3 | Planner 与 Verifier 不需要顶级模型，控成本 |
| 检索存储 | 本地 `.pt` + 内存 MaxSim | 数据量级不需要向量数据库 |
| 后端 | FastAPI | 标准选型 |
| 前端 | Gradio | 一周内完成，重展示而非交互 |

## 5. 评测

自建评测集：3 份 A 股上市公司年报（贵州茅台、宁德时代、比亚迪），手工标注 30 个 QA 对，分三类：

- **L1 单页事实型**（10 题）：可直接从一页定位答案
- **L2 跨页/表格图表型**（10 题）：需要解读表格或图表
- **L3 跨文档对比型**（10 题）：需要规划 + 多次检索

| 方案 | L1 | L2 | L3 | Overall |
|---|---|---|---|---|
| 文本 RAG（bge-m3 + chunk）| __ | __ | __ | __ |
| ColPali 检索 + 单轮 LLM | __ | __ | __ | __ |
| **FinDoc Agent (ours)** | __ | __ | __ | __ |

> 评测脚本与标注数据见 `eval/`。
> ⚠️ 评测集规模有限（30 题），结论用于工程方案对比，不作为通用基准。

## 6. 项目结构

```
findoc-agent/
├── agent/
│   ├── graph.py              # LangGraph 状态机定义
│   ├── nodes/
│   │   ├── planner.py
│   │   ├── executor.py
│   │   ├── verifier.py       # Reflexion 核心
│   │   └── synthesizer.py
│   ├── state.py              # State 数据结构
│   └── prompts/              # 各节点的 prompt 模板
├── tools/
│   ├── colpali_tool.py       # 封装自微调 ColPali
│   ├── vlm_tool.py
│   └── calculator.py
├── ingestion/
│   ├── pdf_to_pages.py
│   └── build_index.py        # ColPali embedding 离线构建
├── eval/
│   ├── qa_dataset.jsonl
│   └── run_eval.py
├── app/
│   └── gradio_app.py
└── README.md
```

## 7. 快速开始

```bash
# 1. 环境
conda create -n findoc python=3.10 && conda activate findoc
pip install -r requirements.txt

# 2. 构建索引（首次运行）
python ingestion/build_index.py --pdf_dir ./data/reports

# 3. 启动
python app/gradio_app.py
```

## 8. 关键设计决策记录（Design Decisions）

为了便于复盘与面试讨论，以下记录几个关键的工程取舍：

1. **为什么不直接 fork Dify / FinRobot？** 两者目标都是平台化，与本项目"垂直领域 + 视觉检索深度集成"的定位不符；从零搭建反而更易讲清每一行代码的设计意图。
2. **为什么 Planner 用 decomposition 而非 ReAct？** 金融问答的子任务边界清晰，先规划再执行更稳定且 token 更省；ReAct 在该场景下容易陷入"反复检索同一页"。
3. **为什么 Verifier 只校验充分性 + 一致性，不做完整 self-critique？** 通用 self-critique 容易让 LLM 自我说服；限定为"客观可判定"的两类校验更可靠。
4. **为什么不引入向量数据库？** 当前数据规模下内存索引已足够，且 ColPali 的 multi-vector 在主流向量库中支持参差，引入反而成为瓶颈。

## 9. 局限性与后续工作

- 评测集规模偏小（30 题），后续计划扩展到 200+ 题并引入 LLM-as-judge
- 当前不支持多模态输出（如生成对比图表），仅文本 + 引用
- ColPali 的 multi-vector 存储未压缩，长期需引入 token pooling 或 binary quantization

## 10. References

- ColPali: Faysse et al., *Efficient Document Retrieval with Vision Language Models*, 2024
- Reflexion: Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS 2023
- LangGraph: https://langchain-ai.github.io/langgraph/
