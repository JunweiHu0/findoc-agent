# BUGS

> 代码审查发现的问题清单。按"是否真 bug"分级：🔴 真实 bug（建议尽快修）/ 🟡 设计或边界问题 / 🟢 小事项。

---

## 🔴 真实 bug（单用户测试碰不到，生产会暴露）

### B1. 进度 hook 是模块级全局，并发请求互相串流

**位置**：`tools/colpali_tool.py:42` `_progress_hook` + `backend/server.py:194` `set_colpali_hook`

**现象**：模块级单例。两个并发请求各自调 `set_progress_hook(closure)`，后者覆盖前者。A 用户的 executor 触发 `_report_progress(...)` 走的是 B 的 closure → 推到 B 的 SSE 流。A 看到 B 的进度，B 看到 A 的。

**修法**：用 `contextvars.ContextVar`，配合 `contextvars.copy_context().run(...)` 在线程入口隔离。

---

### B2. Qdrant Point ID 用 Python `hash()`，进程间不稳定

**位置**：`ingestion/push_to_qdrant.py:69`

```python
point_id = abs(hash(f"{doc_id}__p{page_num}")) & 0x7FFFFFFFFFFFFFFF
```

**现象**：Python `hash(str)` 默认随机化（PYTHONHASHSEED 每次进程启动不同）。重跑 `push_to_qdrant` 时同一份文档同一页生成的 ID 不同 → 不再 upsert，而是 insert 重复点。每次部署/容器重启/手动重跑后 collection 都翻倍增长。

**修法**：用 `hashlib.md5(...).digest()[:8]` 取确定性整数，或直接用字符串 ID `f"{doc_id}__p{page_num}"`（Qdrant 1.7+ 支持）。

---

### B3. 模型懒加载无锁，并发首请求会双倍加载 OOM

**位置**：`tools/colpali_tool.py:79` `_ensure_model_loaded`

**现象**：`if _state["model"] is not None: return` 之后无锁加载。两个线程同时通过检查 → 同时跑 `_load_model()` → GPU 显存翻倍。3060 6GB 上 ColQwen2 bf16 ~5GB，第二份必 OOM。

**当前规避**：startup 预加载已预热单 worker。但配了 `colqwen_url` 而服务挂了 → 并发请求同时 fallback 本地仍会触发。

**修法**：double-checked locking with `threading.Lock`。

---

### B4. `resolve_path` 用 CWD 而非项目根

**位置**：`ingestion/model_loader.py:19`

**现象**：`config.yaml` 里的 `model_name: ./models/colqwen2-v0.1` 是相对项目根的，但 `resolve_path` 只看 `Path.cwd()`。从其他目录启动服务（systemd unit、容器 entrypoint、`cd /tmp && python -m ...`）会找不到模型抛 `OSError`。

**修法**：基于 `agent.config.ROOT` 而非 CWD。

---

### B5. `build_index` 幂等检查不验证完整性

**位置**：`ingestion/build_index.py:66`

```python
if pt_path.exists() and meta_path.exists():
    return 0  # skip
```

**现象**：上次跑到一半被 Ctrl+C / OOM，`embeddings.pt` 已写但页数不全。下次运行直接跳过，**用损坏索引检索一辈子**。

**修法**：先写 `.tmp` 再 `Path.rename()`（POSIX 原子），或加 `.done` sentinel 文件。

---

## 🟡 设计或边界问题（影响正确性/性能/可维护性）

### B6. `max_reflexion_iter=3` 实际只允许 2 次反思

**位置**：`agent/graph.py:16` + `agent/nodes/verifier.py:15`

**现象**：verifier 进入即 `iter+=1`，graph 路由 `>= 3` 时强制 synthesizer。第 3 次 verifier 的判断结果被丢弃。文档说"反思 3 次"语义偏差。

**修法**：把路由阈值改为 `>` 或文档改成"verifier 调用最多 3 次"。

---

### B7. Qdrant collection 为空时不 fallback

**位置**：`tools/colpali_tool.py:238`

**现象**：只在抛异常时回退到 in_memory。Qdrant 服务起来了但 collection 是空的（用户开了 docker compose 但忘了 `push_to_qdrant`）→ 返回 `[]` 不抛异常 → 不回退 → agent 反思 3 轮空答案。

**修法**：空结果时也 fallback；或启动时 collection 大小检查 + warning。

---

### B8. Executor VLM 调用串行

**位置**：`agent/nodes/executor.py:40`

**现象**：top_k=5 串行 ≈ 10s（已知 todo）。

**修法**：`ThreadPoolExecutor(max_workers=5).map(...)`。

---

### B9. Verifier 反思只追加 1 个 SubTask

**位置**：`agent/nodes/verifier.py:43`

**现象**：`missing_info` 是字符串，可能描述"需要 X 和 Y"两段信息，被打包成单个 sub_query 一次检索。召回质量打折。

**修法**：让 verifier 输出 `missing_info: list[str]`，逐条 append。

---

### B10. `chat_history` 字段定义但未使用

**位置**：`agent/state.py:55`

**现象**：4 个节点都不读不写，对话刷新即丢。无法支持指代追问。已知 todo。

---

### B11. MOCK_HITS 在生产可能误导

**位置**：`tools/colpali_tool.py:31` + `_in_memory_retrieve` 索引空时返回 mock

**现象**：生产部署索引文件被误删 → mock 命中 → VLM 读不存在的图返回 mock 文本 → 用户拿到一本正经的假答案。无报错。

**修法**：非测试环境直接抛 `RuntimeError`，或加 env gate `FINDOC_ALLOW_MOCK=1`。

---

### B12. calculator 不限制指数大小

**位置**：`tools/calculator.py:19` `ast.Pow`

**现象**：LLM 生成 `2**100000` 卡 CPU、占内存。Planner prompt 虽限制但无法 100% 防 prompt injection。

**修法**：Pow 时检查 `right.value <= 100`。

---

### B13. FastAPI `@app.on_event("startup")` 已 deprecated

**位置**：`backend/server.py:160`

**现象**：FastAPI 0.110+ 推荐 `lifespan` 上下文管理器，未来版本会失效。当前只是 deprecation warning。

**修法**：迁移到 `@asynccontextmanager` + `FastAPI(lifespan=...)`。

---

## 🟢 小事项

- **B14**：`requirements.txt` 仍声明 `gradio>=4.40.0`，但 `app/gradio_app.py` 已被删（DEVLOG 标记弃维），可移除依赖
- **B15**：`encode_pages` 每个 batch 都调 `torch.cuda.empty_cache()`（`model_loader.py:80`），循环结束调一次即可
- **B16**：`schemas.py` 的 `expected_output_schema` 是 `str` 而非 `Literal["number","text","table"]`，LLM 偶尔输出 `numeric` 之类变体不会被校验拦下
- **B17**：`_load_doc_metadata` 每次 plan 都同步读磁盘（`agent/nodes/planner.py:17`），可加 mtime 缓存

---

## 修复优先级建议

| 顺序 | 项 | 理由 |
|---|---|---|
| 1 | B1 进度 hook | 多用户必触发，UX 直接错乱 |
| 2 | B2 Qdrant Point ID | 重启即数据腐化，越用越糟 |
| 3 | B3 模型加载锁 | fallback 路径下并发会 OOM 服务 |
| 4 | B5 索引幂等性 | 静默数据腐化，最难发现 |
| 5 | B4 resolve_path | 部署到容器/systemd 必踩 |
| 6 | B11 MOCK 兜底 | 生产隐性假数据风险 |
| 7+ | 其余 | 性能 / 边界 / 可维护性 |
