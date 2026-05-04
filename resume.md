# FinDoc Agent · 简历文案

## 项目名称

**FinDoc Agent · 面向金融文档的多模态 RAG Agent 系统**

从视觉检索、Agent 状态机到后置审计的完整 Agent 系统。

---

## 正文（按岗位选一版）

### 版本 A：面 Agent / AI 应用岗

> 基于 LangGraph 的多模态 RAG Agent，面向金融年报的视觉检索与跨文档推理。
>
> - **视觉检索绕过 OCR**：ColQwen2 多向量 MaxSim 直接检索页面图像，保留表格/图表布局。Qdrant 服务端 MaxSim。
> - **7 节点状态机**：Planner → Executor → Verifier（结构化 MissingFact，4 类 root_cause 诊断）→ Remediation（按根因差异化修复）→ Synthesizer → Grounding（引用/数值后置审计，规则非 LLM）。
> - **关键决策**：Planner 用 decomposition 而非 ReAct；Verifier 输出结构化 schema 而非自由文本；Grounding 用规则校验避免 LLM 自身幻觉。
> - **跨轮记忆**：`(entity, period, metric, value)` 结构化抽取 + SQLite，追问检索调用降 40–60%。
> - **技术栈**：Python · LangGraph · FastAPI SSE · Qdrant · ColQwen2 · DeepSeek V4 · Qwen VLM

### 版本 B：面全栈 / 工程岗

> 独立搭建面向金融文档的多模态 RAG Agent 全栈系统。
>
> - **检索引擎**：ColQwen2 多向量 + Qdrant，4 种部署组合（本地/远程 × 内存/Qdrant），异常自动 fallback。
> - **Agent 核心**：LangGraph 7 节点状态机，结构化根因诊断 + 差异化 reflexion + 后置引用审计。Tool Registry + pydantic dispatch 可扩展工具调度。
> - **后端**：FastAPI SSE 流式推送节点进度 + token 输出。后台线程 + queue 解决 LangGraph astream 阻塞。启动期预加载消冷启动。
> - **数据**：PDF → 页面图 → ColQwen2 编码 → .pt / Qdrant upsert 全流水线。VLM 缓存 + 用户上传知识库 + SQLite 对话持久化。
> - **技术栈**：Python · LangGraph · FastAPI · Chainlit · Qdrant · Litserve · SQLite

---

## 每条背后的面试逻辑

| 你写的内容 | 面试官读出的信息 |
|---|---|
| "7 节点状态机" | 理解 Agent 不是单次问答链，是状态流转 + 循环 + 条件路由 |
| "Planner 用 decomposition 而非 ReAct" | 不是调个 LangChain 就完了，知道两种范式区别，有选型判断 |
| "Verifier 输出结构化 MissingFact 而非自由文本" | 知道非结构化输出对下游是灾难，有 schema 设计意识 |
| "remediation 按 4 类 root_cause 分派" | 理解"诊断"和"修复"应该是不同路径，不是一刀切 |
| "Grounding 用规则而非 LLM" | 知道 LLM 的边界在哪——能不用的时候坚决不用 |
| "4 种部署组合 + fallback" | 有系统设计 sense，不是本地跑通就行 |
| "后台线程 + queue 轮询解 astream 阻塞" | 遇到真实工程问题，动手解决了 |
| "模型启动期预加载消冷启动" | 考虑了延迟和用户体验 |

核心原则：**写决策，不写功能**。写"为什么选 A 不选 B"，不写"我做了 A"。面试官看简历只看你有没有判断力——功能列表 AI 也能写。

---

## 个人简介一句话

> 对 Agent 设计的理解：LLM 负责"理解"（语义判断），规则/状态机负责"执行"（确定性分派），边界清晰才能可靠。Planner-Verifier-Grounding 三重校验是降低幻觉的工程手段。

---

## 面试叙述线

按 LEARNLOG §10 的 5 处诊断 → §11 的 7 节点升级 → DEVLOG 的工程决策这条线讲：

1. 旧架构 4 节点有什么弱点（盲规划、修复同质化、零校验……）
2. 针对性地加了 3 个节点 + 重构了 Verifier + 引入了 Tool Registry + 结构化 Fact
3. 每个设计决策有明确的 trade-off（为什么 decomposition 而非 ReAct？为什么规则而非 LLM？为什么向后兼容？）

这条叙事线比罗列技术栈有力十倍——它在展示"你如何思考系统设计"，而非"你用过什么工具"。

---

## 关键词标签

`LangGraph` `Multi-Vector Retrieval` `ColQwen2` `MaxSim` `Vision-Language Model` `RAG Agent` `Reflexion` `Root-Cause Diagnosis` `Tool Registry` `FastAPI SSE` `Qdrant` `Grounding`
