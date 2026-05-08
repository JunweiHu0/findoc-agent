# FinDoc Agent · 简历文案

> 同一项目，三种粒度。投递时按 JD 抓取对应版本；面试时按"叙述线"展开。
> 核心原则：**写决策，不写功能**。

---

## 一行版（朋友圈 / JD 摘要 / GitHub bio）

**FinDoc Agent — 基于 LangGraph 8 节点状态机的金融文档多模态 RAG Agent，ColQwen2 视觉检索 + LLMCompiler 风格 DAG 执行 + 结构化反思 + 三层语义记忆。**

---

## 简历正文（三段式，~ 250 字，建议主用）

> **FinDoc Agent · 面向金融年报的视觉检索 RAG Agent**　·　个人项目　·　Python / LangGraph / FastAPI / Qdrant
>
> - **视觉检索绕过 OCR**：以 ColQwen2 多向量 + MaxSim 直检页面图像，保留表格/图表/数字精度；ViDoRe 上较 ColPali nDCG@5 提升 ~8pp（89% vs 81%）。设计 4 种部署组合（本地/远程 ColQwen × 内存/Qdrant MultiVector），Qdrant 异常自动 fallback，Agent 永不因向量库故障崩溃。
> - **8 节点 LangGraph 状态机**：query_router（关键词+LLM 智能路由，闲聊直答跳过整条检索流水线）→ retrieval_scout（轻量 MaxSim top-3 候选文档）→ planner（两段式：query_class 分类 → 变体 prompt + few-shot 生成 DAG 计划，含 `$tN.value` 跨任务占位符）→ executor（DAG 拓扑分层 ThreadPool 并发 + 跨轮 known_facts 硬命中跳过检索+VLM）→ plan_critic（信号词/失败任务触发按需修订，cursor+iter 重入保护）→ verifier（数值/对比类查询 3 实例并行多数表决）→ remediation（按 4 类根因分派显式 tool_calls 差异化修复 + 三重预算降级保护）→ synthesizer（SSE 流式 token + 正则解析 `[doc p.N]` 引用并剥离虚构项）。
> - **LLMCompiler 跨任务数据流**：DAG plan 中通过 `$tN.value` 占位符实现任务间数值传递；executor 在调用 calculator 前链式解析占位符（ComputedValue → Fact.value → Fact.text 兜底）；synthesizer prompt 硬约束"compute: 行的 value 是真值，禁止心算"。Few-shot 注入 3 个 DAG 示例（毛利率两阶段、跨公司对比、同比增速）。
> - **结构化反思 + 根因诊断**：verifier 输出 pydantic `MissingFact` 含 `retrieval_miss / reading_miss / ambiguous_query / inconsistency` 4 类根因；remediation 按根因构造显式工具调用（`read_page_with_vlm` / `disambiguate_caliber`，不再是字符串前缀污染）；三重预算防死循环（iter=3 / retrieval=10 / vlm=20）。
> - **三层语义记忆**：工作记忆（fact_index dict）→ 情景记忆（conv_facts + ColQwen 128d float16 embedding，cosine ≥ 0.85 硬命中跳过检索，0.5–0.85 软命中作检索先验）→ 语义记忆（hit_count ≥ 3 且 grounding_verified=1 晋升 global_facts 跨对话复用）。embedding 复用 ColQwen 文本编码器零额外依赖，模型未加载回退字符 n-gram。
> - **可扩展能力体系**：工具注册表（ToolSpec 自描述 params/output schema，planner prompt 自动发现，新工具一行 `register`）+ YAML 技能注册表（trigger 关键词 O(1) 匹配 → 注入 plan_template / strategy / verifier 变体，跨 planner/executor/verifier 一致策略）。
> - **健壮性与可观测性**：tenacity 指数退避（瞬时错误重试，401/400 立即失败避免烧 API 额度）+ 结构化 `error_log` + 运行时 `todo_items`（parent_id 串联重试链）+ TokenBudget 触发的上下文压缩（chat_history 正则摘要、evidence 按 entity/period/metric 去重）+ 引用校验正则解析后写回情景记忆驱动语义晋升。
> - **工程交付**：FastAPI SSE 流式后端（event: status/token/node/todo/error/done 全部带 type 字段，前端按 `event:` 行主分派）+ Chainlit 前端（零 agent 代码耦合，纯 HTTP/SSE 边界）+ Qdrant Docker + Litserve ColQwen GPU 服务；已索引 14+ 份年报，30 题评测集覆盖 L1 单事实 / L2 单文档计算 / L3 跨文档对比三档复杂度。
>
> **技术栈**：Python · LangGraph · FastAPI SSE · Qdrant MultiVector · ColQwen2 + LoRA · DeepSeek API · Qwen VLM · Litserve · Chainlit · SQLite WAL · tenacity · pydantic

---

## 极简版（适合空间紧张的简历，~ 4 bullet）

> **FinDoc Agent · 金融文档多模态 RAG Agent**　·　个人项目
>
> - 用 ColQwen2 多向量 + MaxSim 直检页面图像绕过 OCR，保留表格/图表精度；4 种部署组合 + Qdrant 异常自动 fallback。
> - LangGraph 8 节点状态机（智能路由→探查→DAG 规划→分层并发执行→计划评审→并行表决验证→根因驱动修复→流式合成含引用校验），LLMCompiler 风格 `$tN.value` 跨任务数据流 + 结构化反思 + 4 类根因诊断驱动显式工具调用 + 三重预算控制。
> - 三层语义记忆（工作/情景/语义）：cosine 硬命中跳过检索，hit_count + grounding_verified 双条件晋升跨对话；embedding 复用 ColQwen 编码器零依赖。
> - 工具/技能双注册表（自描述 + YAML 配置）+ tenacity 指数退避 + TokenBudget 自动压缩；FastAPI SSE + Chainlit 前后端零耦合。

---

## 面试逻辑映射（一句话 → 面试官心里的 OS）

| 简历描述 | 面试官读出的信号 |
|---|---|
| "decomposition 而非 ReAct" | 知道两种范式区别，做过选型判断而不是抓个框架就用 |
| "ColQwen2 多向量 + MaxSim 绕过 OCR" | 不是把 LangChain 拿来包一下，理解 RAG 的真问题在哪 |
| "8 节点状态机，显式条件边" | 要可观测、可控、可审计；理解黑盒循环的代价 |
| "query_router 跳过非检索轮次" | 知道不是所有查询都需要跑全流水线；有成本意识 |
| "LLMCompiler DAG + $tN.value 跨任务数据流" | 不是把 plan 变成 list 跑串行；理解编译器思想的 DAG 调度 |
| "plan_critic 按需触发 + cursor/iter 重入保护" | 知道什么时候该重新规划，不是无脑迭代，且有防振荡机制 |
| "4 类根因驱动显式工具调用 + 预算降级" | 诊断和修复解耦；不再用字符串前缀 hack 而是真正的工具调用 |
| "3 实例并行多数表决" | 知道单点 LLM 输出不可靠，引入冗余校验 |
| "正则解析引用 + 集合对比剥离虚构引用，零 LLM" | 知道 LLM 边界在哪，能不用 LLM 的时候坚决不用 |
| "三层记忆 + cosine 硬命中跳过检索 + 晋升机制" | 不满足于每次重新搜，有完整的记忆系统设计 |
| "工具/技能双注册表" | 写过框架代码，理解扩展性 vs 易用性的权衡 |
| "tenacity 区分瞬时/致命错误" | 不是 try/except 包一切；知道 401 重试是在烧钱 |
| "4 种部署组合 + 自动 fallback" | 有系统健壮性意识，不是本地跑通就行 |
| "ColQwen 文本编码器复用做 embedding" | 资源约束意识；不为 episode memory 引入新依赖 |
| "前后端零耦合，前端按 SSE event: 行主分派" | 有清晰的边界设计感，不是把所有东西堆在一个进程 |

---

## 面试叙述线（按这个顺序讲，10 分钟讲完）

**1. 起点：问题观察**

> "金融年报的表格、图表、脚注，OCR 解出来结构全乱。传统 RAG 从 PDF 抽文本切块这步就丢了一半信息。"

**2. 第一层选型：ColQwen2 + MaxSim**

> "所以我用 ColQwen2 直接编码页面图像，每页 ~1024 个 patch 向量，每维 128。检索用 MaxSim — query 每个 token 找最匹配的 patch。"
> （此处可拓展讲为什么用 Qdrant MultiVector 而不是 Chroma/Pinecone — 它们只支持 single-vector）

**3. 第二层选型：LangGraph decomposition 而非 ReAct**

> "金融问题往往是多步对比 — '比较茅台和宁德 2023 三年毛利率'。ReAct 的 think-act 循环对这种结构化任务不可控。我用 LangGraph 显式状态机 — 8 个节点，每个节点 I/O 可观测，反思循环用条件边管控。"

**4. 第三层：智能入口 + DAG 执行**

> "不是所有查询都需要跑全流水线。query_router 用关键词+轻量 LLM 判断——'你好'直接答，'茅台 2023 营收'走检索。需要检索时，planner 产出 DAG plan，任务间通过 `$tN.value` 占位符传递数值——比如 task_1 算出营收再传给 task_3 算毛利率。executor 拓扑排序 + 同层 ThreadPool 并发，plan_critic 在信号词或失败时按需修订 plan。"

**5. 第四层细节：结构化反思**

> "verifier 的输出不是自由文本 '我觉得证据不够'，是 pydantic schema 的 MissingFact[] — 每条带 root_cause 枚举（4 类）。remediation 按根因构造显式工具调用 — retrieval_miss 放宽 top_k 重检索，reading_miss 构造 read_page_with_vlm 重读同批页，inconsistency 触发 disambiguate_caliber。三重预算防死循环。"

**6. 第五层：引用校验**

> "Synthesizer 出答案后，从文本正则解析 `[doc p.N]` 引用，对比 evidence 集合——虚构引用直接剥离，只把模型真正用到的页返给前端。纯规则，零 LLM 调用。"

**7. 第六层：记忆和规模**

> "三层记忆 — 工作 dict、情景 cosine 检索、语义跨对话晋升。embedding 直接复用 ColQwen 文本编码器，省一个嵌入模型依赖。"

**8. 收：工程交付**

> "前后端零耦合 — Chainlit 只通过 SSE 消费节点流，agent 代码完全不知道前端存在。Qdrant 远程 ColQwen 任何一环挂掉都自动降级到本地 + 内存。"

---

## 量化指标速查（面试问"有什么数字"时立刻报）

| 维度 | 数字 |
|---|---|
| 索引文档数 | 14+ 份年报 |
| 评测集规模 | 30 题，覆盖 L1 / L2 / L3 三档复杂度 |
| 状态机节点数 | 8（7 常驻 + 1 按需 plan_critic） |
| 根因分类 | 4 类（驱动 4 种修复策略） |
| 并行验证实例 | 3（strict / base / numeric） |
| DAG few-shot 示例 | 3（毛利率两阶段 / 跨公司对比 / 同比增速） |
| 部署组合 | 4 种（本地/远程 ColQwen × 内存/Qdrant） |
| 工具数 / 技能数 | 4 / 4，均可一行扩展 |
| 反思预算 | iter ≤ 3，retrieval ≤ 10，vlm ≤ 20 |
| plan_critic 上限 | max 2 次修订 |
| 记忆层数 | 3（工作 / 情景 / 语义） |
| 硬命中阈值 | cosine ≥ 0.85 跳过检索，0.5–0.85 作先验 |
| 晋升阈值 | hit_count ≥ 3 且 grounding_verified=1 |
| 引用校验方式 | 纯正则 + 集合查，零 LLM 调用 |
| 检索精度对比 | ColQwen2 ViDoRe nDCG@5 ~89% vs ColPali ~81% |
| 模型尺寸约束 | ColQwen2-2B bf16 适配 RTX 3060 6GB |

---

## 技术关键词（ATS / GitHub topics）

`LangGraph` `Multimodal-RAG` `ColQwen2` `MaxSim` `Late-Interaction` `Qdrant-Multivector`
`LLMCompiler` `DAG-Scheduling` `Reflexion` `Root-Cause-Diagnosis` `Parallel-Voting`
`Tool-Registry` `Skill-Registry` `Semantic-Memory` `Episodic-Memory` `Memory-Promotion`
`Citation-Verification` `FastAPI-SSE` `Chainlit` `Litserve` `Tenacity` `Pydantic`

---

## 写法 checklist（自检用）

- [ ] 每条 bullet 都能让面试官追问"为什么这么做"，而不是"做了什么"
- [ ] 凡是有数字的地方都写了数字（节点数 / 阈值 / 预算 / 文档数 / 评测题数）
- [ ] 出现了对比基线（ColQwen2 vs ColPali、decomposition vs ReAct、Qdrant vs Chroma、LLMCompiler DAG vs 扁平 list）
- [ ] 出现了"做出来 vs 没做"的取舍信号（query_router vs 全量检索、root cause vs 自由文本、规则审计 vs LLM 审计、复用编码器 vs 新依赖、显式工具调用 vs 字符串前缀 hack）
- [ ] 没有"successfully implemented" 这种零信息词
- [ ] 一行版能塞进 GitHub bio，三段版能塞进 1 页简历，叙述线能撑过 10 分钟面试
