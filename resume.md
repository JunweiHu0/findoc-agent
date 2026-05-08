# FinDoc Agent · 简历文案

> 同一项目，三种粒度。投递时按 JD 抓取对应版本；面试时按"叙述线"展开。
> 核心原则：**写决策，不写功能**。

---

## 一行版（朋友圈 / JD 摘要 / GitHub bio）

**FinDoc Agent — 基于 LangGraph 8 节点状态机的金融文档多模态 RAG Agent，ColQwen2 视觉检索 + 结构化反思 + 三层语义记忆。**

---

## 简历正文（三段式，~ 250 字，建议主用）

> **FinDoc Agent · 面向金融年报的视觉检索 RAG Agent**　·　个人项目　·　Python / LangGraph / FastAPI / Qdrant
>
> - **视觉检索绕过 OCR**：以 ColQwen2 多向量 + MaxSim 直检页面图像，保留表格/图表/数字精度；ViDoRe 上较 ColPali nDCG@5 提升 ~8pp（89% vs 81%）。设计 4 种部署组合（本地/远程 ColQwen × 内存/Qdrant MultiVector），Qdrant 异常自动 fallback 到内存检索，agent 永不因向量库故障崩溃。
> - **8 节点 LangGraph 状态机**：retrieval_scout 预检索 top-3 候选 → planner（两段式：轻量分类 → 变体 prompt + few-shot 生成 DAG 计划）→ executor（**ThreadPool 拓扑分层并发** + 跨轮缓存硬命中跳过检索）→ plan_critic（信号词/任务失败触发的按需计划修订）→ verifier（多步/对比类查询启用 **strict/base/numeric 三实例并行多数表决**）→ remediation（按 4 类根因差异化修复 + 预算降级）→ synthesizer（SSE 流式 token）→ grounding（纯规则后置审计）。
> - **结构化反思 + 根因诊断**：verifier 输出 pydantic schema 的 `MissingFact`，含 `retrieval_miss / reading_miss / ambiguous_query / inconsistency` 4 类根因；remediation 按根因分派差异化修复（重检索 / 重读 / 改写 / 口径消歧），三层预算（max_iter=3 / retrieval=10 / vlm=20）防死循环。
> - **三层语义记忆**：工作记忆（fact_index dict）→ 情景记忆（conv_facts + 128d ColQwen embedding，cosine ≥ 0.85 硬命中跳过检索，0.5–0.85 软命中作检索先验）→ 语义记忆（hit_count ≥ 3 且 grounding_verified=1 晋升 global_facts 跨对话复用）。embedding 复用 ColQwen 文本编码器零额外依赖，模型未加载回退字符 n-gram。
> - **可扩展能力体系**：**工具注册表**（ToolSpec 自描述 params/output schema，planner prompt 自动发现，新工具一行 `register`）+ **YAML 技能注册表**（trigger 关键词 O(1) → 注入 plan_template / strategy / verifier 变体，跨 planner/executor/verifier 一致策略）。
> - **健壮性与可观测性**：tenacity 指数退避（瞬时错误重试，401/400 立即失败避免烧 API 额度）+ 结构化 `error_log` + 运行时 `todo_items`（parent_id 串联重试链）+ TokenBudget 触发的上下文压缩（chat_history 正则摘要、evidence 按 entity/period/metric 去重）+ grounding 纯规则审计反向写回情景记忆驱动语义晋升。
> - **工程交付**：FastAPI SSE 流式后端 + Chainlit 前端（零 agent 代码耦合，纯 HTTP/SSE 边界）+ Qdrant Docker + Litserve ColQwen GPU 服务；已索引 14+ 份年报，30 题评测集覆盖 L1 单事实 / L2 单文档计算 / L3 跨文档对比 三档复杂度。
>
> **技术栈**：Python · LangGraph · FastAPI SSE · Qdrant MultiVector · ColQwen2 + LoRA · DeepSeek API · Qwen VLM · Litserve · Chainlit · SQLite WAL · tenacity · pydantic

---

## 极简版（适合空间紧张的简历，~ 4 bullet）

> **FinDoc Agent · 金融文档多模态 RAG Agent**　·　个人项目
>
> - 用 ColQwen2 多向量 + MaxSim 直检页面图像绕过 OCR，保留表格/图表精度；4 种部署组合 + Qdrant 异常自动 fallback。
> - LangGraph 8 节点状态机（探查→规划→DAG 并发执行→计划评审→并行表决验证→根因驱动修复→流式合成→规则审计），结构化反思 + 4 类根因诊断驱动差异化修复 + 三层预算控制。
> - 三层语义记忆（工作/情景/语义）：cosine 硬命中跳过检索，hit_count + grounding_verified 双条件晋升跨对话；embedding 复用 ColQwen 编码器零依赖。
> - 工具/技能双注册表（自描述 + YAML 配置）+ tenacity 指数退避 + TokenBudget 自动压缩；FastAPI SSE + Chainlit 前后端零耦合。

---

## 面试逻辑映射（一句话 → 面试官心里的 OS）

| 简历描述 | 面试官读出的信号 |
|---|---|
| "decomposition 而非 ReAct" | 知道两种范式区别，做过选型判断而不是抓个框架就用 |
| "ColQwen2 多向量 + MaxSim 绕过 OCR" | 不是把 LangChain 拿来包一下，理解 RAG 的真问题在哪 |
| "8 节点状态机" 而非 "AgentExecutor" | 要可观测、可控、可审计；理解黑盒循环的代价 |
| "DAG 拓扑分层并发执行" | 不是把 plan 变成 list 跑串行，理解依赖图调度 |
| "plan_critic 按需触发" | 知道什么时候该重新规划，不是无脑迭代 |
| "4 类根因驱动差异化修复 + 预算降级" | 诊断和修复解耦；理解 LLM 调用是有成本的 |
| "3 实例并行多数表决" | 知道单点 LLM 输出不可靠，引入冗余校验 |
| "纯规则校验引用和数值" | 知道 LLM 边界在哪，能不用 LLM 的时候坚决不用 |
| "三层记忆 + cosine 硬命中跳过检索 + 晋升机制" | 不满足于每次重新搜，有完整的记忆系统设计 |
| "工具/技能双注册表" | 写过框架代码，理解扩展性 vs 易用性的权衡 |
| "tenacity 区分瞬时/致命错误" | 不是 try/except 包一切；知道 401 重试是在烧钱 |
| "4 种部署组合 + 自动 fallback" | 有系统健壮性意识，不是本地跑通就行 |
| "ColQwen 文本编码器复用做 embedding" | 资源约束意识；不为 episode memory 引入新依赖 |
| "前后端零耦合，纯 SSE 边界" | 有清晰的边界设计感，不是把所有东西堆在一个进程 |

---

## 面试叙述线（按这个顺序讲，10 分钟讲完）

**1. 起点：问题观察**

> "金融年报的表格、图表、脚注，OCR 解出来结构全乱。传统 RAG 从 PDF 抽文本切块这步就丢了一半信息。"

**2. 第一层选型：ColQwen2 + MaxSim**

> "所以我用 ColQwen2 直接编码页面图像，每页 ~1024 个 patch 向量，每维 128。检索用 MaxSim — query 每个 token 找最匹配的 patch。"
> （此处可拓展讲为什么用 Qdrant MultiVector 而不是 Chroma/Pinecone — 它们只支持 single-vector）

**3. 第二层选型：LangGraph decomposition 而非 ReAct**

> "金融问题往往是多步对比 — '比较茅台和宁德 2023 三年毛利率'。ReAct 的 think-act 循环对这种结构化任务不可控。我用 LangGraph 显式状态机 — 8 个节点，每个节点 I/O 可观测，反思循环用条件边管控。"

**4. 第三层细节：结构化反思**

> "verifier 的输出不是自由文本 '我觉得证据不够'，是 pydantic schema 的 MissingFact[] — 每条带 root_cause 枚举（4 类）。remediation 按根因分派 — retrieval_miss 就放宽 top_k 重检索，reading_miss 就重读同样的页面但换 VLM 指令，inconsistency 就触发口径消歧工具。三层预算（迭代/检索/VLM）防死循环。"

**5. 第四层：可靠性收尾**

> "Synthesizer 出答案后过 grounding — 纯正则 + 集合查找，零 LLM 调用，验证每个 [doc_id p.N] 引用真实存在、每个数字能在事实里找到 ±0.1% 的匹配。失败的引用直接剥离，不可信整体加置信度横幅。"

**6. 第五层：记忆和规模**

> "三层记忆 — 工作 dict、情景 cosine 检索、语义跨对话晋升。embedding 直接复用 ColQwen 文本编码器，省一个嵌入模型依赖。"

**7. 收：工程交付**

> "前后端零耦合 — Chainlit 只通过 SSE 消费节点流，agent 代码完全不知道前端存在。Qdrant 远程 ColQwen 任何一环挂掉都自动降级到本地 + 内存。"

---

## 量化指标速查（面试问"有什么数字"时立刻报）

| 维度 | 数字 |
|---|---|
| 索引文档数 | 14+ 份年报 |
| 评测集规模 | 30 题，覆盖 L1 / L2 / L3 三档复杂度 |
| 状态机节点数 | 8（含按需触发的 plan_critic） |
| 根因分类 | 4 类（驱动 4 种修复策略） |
| 并行验证实例 | 3（strict / base / numeric） |
| 部署组合 | 4 种（本地/远程 ColQwen × 内存/Qdrant） |
| 工具数 / 技能数 | 4 / 4，均可一行扩展 |
| 反思预算 | iter ≤ 3，retrieval ≤ 10，vlm ≤ 20 |
| 记忆层数 | 3（工作 / 情景 / 语义） |
| 硬命中阈值 | cosine ≥ 0.85 跳过检索，0.5–0.85 作先验 |
| 晋升阈值 | hit_count ≥ 3 且 grounding_verified=1 |
| 数值审计容差 | ±0.1% |
| 检索精度对比 | ColQwen2 ViDoRe nDCG@5 ~89% vs ColPali ~81% |
| 模型尺寸约束 | ColQwen2-2B bf16 适配 RTX 3060 6GB |

---

## 技术关键词（ATS / GitHub topics）

`LangGraph` `Multimodal-RAG` `ColQwen2` `MaxSim` `Late-Interaction` `Qdrant-Multivector`
`Reflexion` `Root-Cause-Diagnosis` `Parallel-Voting` `DAG-Scheduling`
`Tool-Registry` `Skill-Registry` `Semantic-Memory` `Episodic-Memory`
`Grounding-Audit` `FastAPI-SSE` `Chainlit` `Litserve` `Tenacity` `Pydantic`

---

## 写法 checklist（自检用）

- [ ] 每条 bullet 都能让面试官追问"为什么这么做"，而不是"做了什么"
- [ ] 凡是有数字的地方都写了数字（节点数 / 阈值 / 预算 / 文档数 / 评测题数）
- [ ] 出现了对比基线（ColQwen2 vs ColPali、decomposition vs ReAct、Qdrant vs Chroma）
- [ ] 出现了"做出来 vs 没做"的取舍信号（root cause vs 自由文本、规则审计 vs LLM 审计、复用编码器 vs 新依赖）
- [ ] 没有"successfully implemented" 这种零信息词
- [ ] 一行版能塞进 GitHub bio，三段版能塞进 1 页简历，叙述线能撑过 10 分钟面试
