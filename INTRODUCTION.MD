# FinDoc Agent — 目录结构与代码说明

```
findoc-agent/
├── .claude/                        # Claude Code 本地配置
│   └── settings.local.json         #   项目级权限/钩子设置（非代码）
│
├── .git/                           # Git 仓库元数据（非代码）
│
├── agent/                          # Agent 核心：LangGraph 状态机 + 各节点实现
│   ├── __init__.py                 #   包标记（空）
│   ├── config.py                   #   全局配置加载：读 config.yaml + .env，解析路径，导出常量
│   ├── state.py                    #   AgentState TypedDict 定义 + SubTask/PageHit/Fact 等 Pydantic 数据模型
│   ├── schemas.py                  #   LLM 结构化输出目标的 Pydantic schema（PlannerOutput / VerifierOutput）
│   ├── llm.py                      #   LLM 工厂：按 role（planner/verifier/synthesizer）创建 ChatOpenAI 实例，统一走 DeepSeek API
│   ├── graph.py                    #   LangGraph 状态机装配：4 个节点 + 1 个条件路由 + reflexion 预算控制
│   ├── prompts/                    #   Prompt 模板目录
│   │   ├── __init__.py             #     load_prompt(name) 工具函数：从 .txt 文件读入模板字符串
│   │   ├── planner.txt             #     Planner 节点 prompt：要求拆分用户问题为有序子任务 JSON
│   │   ├── verifier.txt            #     Verifier 节点 prompt：判断证据充分性 + 一致性，输出 JSON
│   │   └── synthesizer.txt         #     Synthesizer 节点 prompt：汇总证据生成带内联引用的最终答案
│   └── nodes/                      #   LangGraph 节点实现
│       ├── __init__.py             #     包标记（空）
│       ├── planner.py              #     Planner 节点：LLM 拆解任务 → list[SubTask]；失败时 fallback 单步执行
│       ├── executor.py             #     Executor 节点：按 cursor 消费 plan，number → calculator，其他 → ColPali + VLM
│       ├── verifier.py             #     Verifier 节点：LLM 判断充分性/一致性；不充分时追加 sub-task；plan 耗尽时启发式 fallback
│       └── synthesizer.py          #     Synthesizer 节点：汇总 facts + computed_values，LLM 生成带引用答案
│
├── tools/                          # Agent 可调用的三个工具
│   ├── __init__.py                 #   包标记（空）
│   ├── colpali_tool.py             #   ColQwen2/ColPali 多向量视觉检索：query 编码 + MaxSim + top-k 排序；有 mock fallback
│   ├── vlm_tool.py                 #   页面 VLM 解读：OpenAI-compat vision API（默认 Qwen Dashscope），图 → 结构化文本；有 mock fallback
│   └── calculator.py               #   安全数值计算器：AST 白名单，只允许 +-*/ // % **，拒绝任意函数调用和变量名
│
├── ingestion/                      # 离线数据管线：PDF → 页面图像 → ColQwen2 索引
│   ├── __init__.py                 #   包标记（空）
│   ├── pdf_to_pages.py             #   PDF 转 PNG：pdf2image 逐页渲染，中文文件名 → ASCII doc_id 映射，幂等可重入
│   └── build_index.py              #   建索引：ColQwen2 离线编码所有页面 → embeddings.pt + meta.json + doc_memory.json
│
├── eval/                           # 评测模块
│   ├── __init__.py                 #   包标记（空）
│   ├── run_eval.py                 #   评测脚本：遍历 qa_dataset.jsonl，跑完整 graph，输出 results.csv
│   └── qa_dataset.jsonl            #   评测数据集（当前 3 题示例，L1/L2/L3 各一）
│
├── app/                            # 应用层
│   ├── __init__.py                 #   包标记（空）
│   └── gradio_app.py               #   Gradio 聊天界面：compile graph → ChatInterface，展示答案 + 引用
│
├── models/                         # 本地模型权重（.gitignore 排除）
│   └── colqwen2-v0.1/              #   ColQwen2 微调适配器（PEFT adapter），含 adapter_config.json 等
│
├── data/                           # 运行时数据（.gitignore 排除）
│   ├── reports/                    #   原始 PDF 年报
│   ├── pages/                      #   渲染后的 PNG 页面（按 doc_id 分子目录）
│   ├── images/                     #   （用途待确认，当前有大量文件）
│   └── index/                      #   离线索引输出（按 doc_id 分子目录，每个含 embeddings.pt + meta.json）
│
├── config.yaml                     # 全局配置文件：agent/llm/retriever/vlm/ingestion 各段参数
├── .env                            # 环境变量（API key 等，.gitignore 排除）
├── .env.example                    # 环境变量模板（占位符，被 git track）
├── .gitignore                      # Git 忽略规则
├── requirements.txt                # Python 依赖清单
├── README.md                       # 项目文档：背景/架构/设计决策/快速开始/局限性
├── DEVLOG.md                       # 开发日志：阶段进度/决策记录/阻塞项/下一步
├── REVIEW.md                       # 代码评审：架构问题/代码缺陷/优先级修复表
└── INTRODUCTION.MD                 # 本文件：目录结构与代码说明
```

---

## 逐文件一句话功能

| 文件 | 一句话 |
|---|---|
| `agent/config.py` | 读 `config.yaml` + `.env`，把所有路径和配置常量统一导出 |
| `agent/state.py` | 定义 `AgentState`（TypedDict）+ `SubTask`/`PageHit`/`Fact`/`ComputedValue`/`Citation` 数据类 |
| `agent/schemas.py` | 定义 `PlannerOutput` 和 `VerifierOutput`——LLM 结构化输出的 JSON schema |
| `agent/llm.py` | `get_llm(role)` 工厂函数，按角色返回配置好的 `ChatOpenAI` 实例 |
| `agent/graph.py` | LangGraph 装配入口：建图 → 加节点 → 加边 → `_route_after_verifier` reflexion 条件路由 |
| `agent/prompts/__init__.py` | `load_prompt(name)` 读 `.txt` 模板文件 |
| `agent/prompts/planner.txt` | Planner 的 system prompt：拆解问题 → `{"plan": [...]}` JSON |
| `agent/prompts/verifier.txt` | Verifier 的 system prompt：判断 `is_sufficient` + `inconsistency` + `missing_info` |
| `agent/prompts/synthesizer.txt` | Synthesizer 的 system prompt：汇总证据 + 内联引用 |
| `agent/nodes/planner.py` | Planner 节点：调 LLM 拆任务，解析失败回退到单步 |
| `agent/nodes/executor.py` | Executor 节点：按 `expected_output_schema` 路由到 calculator 或 colpali+vlm |
| `agent/nodes/verifier.py` | Verifier 节点：调 LLM 判充分性，不充分则往 plan 追加新 sub-task |
| `agent/nodes/synthesizer.py` | Synthesizer 节点：调 LLM 汇总所有证据生成最终答案 + 去重引用 |
| `tools/colpali_tool.py` | 视觉检索核心：加载 ColQwen2 模型 + 索引，query 编码 → MaxSim → top-k 页面 |
| `tools/vlm_tool.py` | OpenAI-compat VLM 调用：base64 编码图 + instruction → 结构化抽取文本 |
| `tools/calculator.py` | AST 白名单安全计算器，只接受纯算术表达式 |
| `ingestion/pdf_to_pages.py` | PDF → 逐页 PNG，中文文件名 → `{company}_{year}` doc_id |
| `ingestion/build_index.py` | 离线建 ColQwen2 多向量索引：编码每页 → `embeddings.pt` + `meta.json` + `doc_memory.json` |
| `eval/run_eval.py` | 遍历评测集跑全图，输出 `results.csv` |
| `eval/qa_dataset.jsonl` | 评测 QA 对（目标 30 题，当前 3 题） |
| `app/gradio_app.py` | Gradio ChatInterface，compile graph 后暴露 `query_fn` 给前端 |
| `config.yaml` | 所有可调参数：模型名/路径/设备/dtype/top_k/反射上限/文档别名 |
| `requirements.txt` | Python 依赖完整清单 + 注释说明 |
| `README.md` | 项目文档（背景/架构/设计决策/局限） |
| `DEVLOG.md` | 工程日志（阶段进度/决策记录/阻塞项） |
| `REVIEW.md` | 代码评审（架构问题/代码缺陷/优先级修复表） |

---

## 数据流向

```
PDF 年报                          用户问题
   │                                 │
   ▼                                 ▼
pdf_to_pages.py              agent/graph.py
   │                                 │
   ▼                                 ▼
页面 PNG  ──→ build_index.py ──→ embeddings.pt     Planner (拆子任务)
                                    │               │
                                    ▼               ▼
                              colpali_tool.py    Executor
                              (MaxSim 检索)        │
                                    │          ┌────┴────┐
                                    ▼          ▼         ▼
                              vlm_tool.py  calculator  colpali
                              (页面解读)   (数值计算)   (再检索)
                                    │          │         │
                                    └──────────┴─────────┘
                                             │
                                             ▼
                                        Verifier
                                      (充分/不充分)
                                             │
                                        ┌────┴────┐
                                    充分 │         │ 不充分 → 回 Executor
                                        ▼         │
                                   Synthesizer    │
                                        │         │
                                        ▼         │
                                Final Answer + 引用
```

核心流程：PDF 离线转图 → 离线建 ColQwen2 索引。在线时，用户问题经 Planner 拆解 → Executor 逐子任务调工具（检索+VLM 或 计算器）→ Verifier 反思（不充分则追加子任务重试）→ Synthesizer 汇总生成带引用的最终答案。
