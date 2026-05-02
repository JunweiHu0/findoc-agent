# FinDoc Agent — Code Review

> 2026-05-02 · 全文通读后的工程点评，按影响面和严重程度排序。

---

## 一、架构与设计

### 1.1 Verifier 的 reflexion 追加机制脆弱

`agent/nodes/verifier.py:43` 把 `missing_info` 直接 append 到 `plan` 末尾，依赖 Executor 的 cursor 自然推进到新位置。这能工作，但：

- **无法区分"原始计划"和"反思补丁"**。如果 Verifier 误判，追加了不必要的 sub-task，没有任何机制回滚。
- **一次 reflexion 只追加一个 sub-task**。若一个子任务暴露多个信息缺口，需要多轮 Executor→Verifier 才能补全，跨文档对比（L3）可能 5+ 轮往返。

**建议**：让 Verifier 一次输出 `list[SubTask]` 而非单个 `missing_info`；或在 State 中增加 `original_plan_len` 标记原始计划边界，便于后续观测和审计。

### 1.2 Executor 每次只消费一个子任务

`agent/nodes/executor.py:13-17` 每次调用只处理 `plan[cursor]`。多轮往返次数 = 原始 plan 长度 + 反思补丁数。每次 LangGraph 节点切换都有状态序列化/反序列化开销。

**建议**：Executor 内部加 mini-loop，连续消费同 `target_doc` 的相邻 plan item，减少节点切换次数。

### 1.3 Document Memory 已落盘但未接入 Planner

`ingestion/build_index.py` 已写入 `data/index/doc_memory.json`，但 Planner 硬编码了：

```python
doc_metadata="(no document memory yet — P2 will provide this)"
```

Planner 在不知道有哪些文档可用的情况下做分解，对于"对比茅台和宁德时代的毛利率"，**无法知道应该检索哪些文档**。这是当前最明显的功能缺口。

### 1.4 Episodic Memory 字段空置

`AgentState` 定义了 `chat_history: list[dict]`，无任何节点读写。追问场景（"那它的研发投入呢？"）会丢失上下文。README §3.3 承诺了此能力但未实现。

---

## 二、代码级问题

### 2.1 `colpali_tool.py` — Mock 数据与真实索引不一致

```python
_MOCK_HITS = [
    PageHit(doc_id="moutai_2023", page_num=42, ...),
    PageHit(doc_id="moutai_2023", page_num=43, ...),
    PageHit(doc_id="catl_2023", page_num=58, ...),
]
```

mock 引用 `p042.png` 等不存在的页面，使得 mock 模式无法跑通 Executor → VLM 链路。Mock 应该是"可运行的替身"，当前是"看起来有数据，跑就挂"。

### 2.2 VLM mock 返回值可能污染 Synthesizer

`tools/vlm_tool.py:28-29` 在找不到图像时返回：

```python
f"[mock vlm] (no image) instruction='{instruction}'"
```

这个字符串作为 `Fact.text` 写入 State。真 LLM Synthesizer 能识别这是 mock，但启发式降级路径会原样输出给用户。

### 2.3 `_encode_query` 强制 cast 到 fp16

`tools/colpali_tool.py:116`：

```python
return emb.to("cpu", dtype=torch.float16)[0]
```

模型以 bf16 推理（配置 `dtype: bfloat16`），bf16 指数范围比 fp16 大。`.to(fp16)` 在极端值上可能产生 inf/nan。应统一为 `model.dtype` 或直接用 bf16。

### 2.4 Calculator 的 NaN 不加区分地流向 Synthesizer

`agent/nodes/executor.py:28-30`：计算失败时 `value=float("nan")`，写入 `ComputedValue`。LLM 可能直接输出 "NaN" 或自行编造数字。

**建议**：失败时写 `value=None, error="reason"`，或触发 Verifier 让 Executor 重新从页面提取数值。

### 2.5 模块级可变单例有线程安全隐患

`tools/colpali_tool.py:36`：

```python
_state: dict = {"model": None, "processor": None, "indexes": None}
```

单线程 Gradio 无问题，但将来换 FastAPI + 多 worker 是竞态条件。建议用类封装或 `functools.lru_cache`。

### 2.6 部分函数缺少类型标注

- `synthesizer.py:_render_evidence(facts, cvs)` — 无参数类型
- `verifier.py:_render_plan(plan)` / `_render_evidence(state)` — 无参数类型
- `executor.py:_run_calculation(sub_task, cursor)` — `sub_task` 无类型

虽然不影响运行，但在 `total=False` 的 TypedDict 上操作时，类型标注可以帮 mypy/pyright 捕获 key 拼写错误。

---

## 三、Prompt 设计

### 3.1 缺少 few-shot 示例

三个 prompt（planner/verifier/synthesizer）都是纯规则描述，无任何输入输出示例。对于结构化 JSON 输出，few-shot 是提升格式遵从度最经济的手段。Planner 解析失败会 fallback 到单步执行——意味着没有 few-shot 的直接代价是**复杂问题因 JSON 格式错误而被降级处理**。

### 3.2 Synthesizer 未强调数值精度

Prompt 只说 "show formulas for any computation"，没有要求保留原始精度。金融场景下 91.52% vs 92% 的差异可能有意义。

### 3.3 Planner prompt 的 `{doc_metadata}` 占位需明确格式

当后续接入真实 `doc_memory.json` 时，`json.dumps` 的格式和 prompt 模板的预期格式需要对齐。建议在 prompt 中给出 doc_metadata 的 schema 说明。

---

## 四、工程实践

### 4.1 零自动化测试 ✅ 已解决

项目没有 `tests/` 目录，各模块靠 `if __name__ == "__main__"` 手动冒烟。缺少：

- Calculator 安全性回归（AST 规则被后续修改打破）
- State reducer（`Annotated[list, add]`）行为验证
- Planner JSON 解析的 fallback 路径

建议至少给 calculator 和 state reducer 加 5-6 个单元测试。

### 4.2 `.env.example` 安全 ✅ 已解决

> 已确认当前 `.env.example` 使用占位符 `tt`，不再含真实 key。

如果之前含 key 的版本已 push 到远程，需要轮换对应的 API key。

### 4.3 评测集仅 3 条

`eval/qa_dataset.jsonl` 含 L1/L2/L3 各 1 题。`run_eval.py` 只跑冒烟，不计算任何指标（准确率、召回率、引用精确率）。README 承诺 30 题待实现。

---

## 五、性能

### 5.1 VLM 逐页串行调用

`executor.py:40-41` 对每个 colpali 命中页串行调用 `vlm_read_page`。top_k=5 × 3 个子任务 = 15 次 API 调用。这是端到端延迟的最大瓶颈。

**建议**：`concurrent.futures.ThreadPoolExecutor` 并行化，或让 VLM 支持单次请求多图输入。

### 5.2 `encode_batch_size: 1` 建索引极慢

2800 页 × ~1s/页 = 约 45 分钟。RTX 3060 6GB 可以开到 batch_size=4~8。

### 5.3 无查询缓存

追问场景下，colpali 对同一文档重复编码 query + 重算 MaxSim。可短期缓存 query embedding 或检索结果。

---

## 总结：优先修复排序

| 优先级 | 问题 | 理由 |
|---|---|---|
| P0 | 接入 doc_memory 到 Planner | 规划功能正确性的前提 |
| P1 | 补 calculator 和 state 的单元测试 | 防回归，投入产出比最高 |
| P1 | 修复 mock 数据与实际索引不一致 | 开发体验，P2 调试阶段尤其重要 |
| P2 | VLM 页面解读并行化 | 端到端延迟最关键瓶颈 |
| P2 | Verifier 支持批量补充 sub-task | 减少 LLM 往返轮次 |
| P2 | `_encode_query` dtype 改为 bf16 或 model.dtype | 避免精度损失 |
| P3 | Episodic memory 多轮对话 | 功能完整性 |
| P3 | Prompt 加 few-shot 示例 | 降低 JSON 解析失败率 |
| P3 | 评测集扩到 30 题 + 自动指标 | 量化迭代方向 |

---

## 总体评价

架构设计清晰（Planner-Executor-Verifier-Synthesizer 分工明确、LangGraph 状态管理干净），README 和 DEVLOG 质量在个人项目中属于上乘——设计决策记录（为什么不用 Dify/ReAct/向量数据库）让读者能准确理解取舍意图。

主要短板在**功能完成度**（P2 未收尾导致 doc memory / mock 数据 / 评测处于"半接入"状态）和**缺乏自动化测试**。这两个在当前单人开发节奏下属正常阶段性问题，按 DEVLOG 计划推进即可。
