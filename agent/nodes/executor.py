"""
Executor node — DAG task scheduler with concurrent tool dispatch / 执行节点——DAG 任务调度 + 并发工具分发。

Builds a dependency graph from plan SubTasks and runs independent tasks at the
same topological level concurrently via ThreadPoolExecutor. Cross-turn fact cache
is checked before retrieval; structured fact extraction runs after every VLM read.
Tool failures write error_log entries; todo_items track per-task runtime status.
构建依赖图，同层无依赖任务并发执行。检索前先查跨轮缓存，VLM 后立即结构化抽取。
工具失败写入 error_log，todo_items 追踪每步运行时状态。
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from loguru import logger

from tools.calculator import calculate
from tools.colpali_tool import colpali_retrieve
from tools.registry import dispatch as _registry_dispatch, REGISTRY
from tools.vlm_tool import vlm_read_page

from ..config import TOP_K
from ..retry import classify_error
from ..state import AgentState, ComputedValue, Fact, TodoItem


# Placeholder syntax — `$<task_id>.value` — used by planner to compose calc steps that
# reference upstream retrieval outputs. Resolved by the executor at dispatch time.
# 占位符语法 $<task_id>.value，由 executor 在分发前解析为字面数字
_PLACEHOLDER_RE = re.compile(r"\$([A-Za-z][\w-]*)\.value")
# Fallback: pull a number from raw fact text when fact.value is None / 当结构化抽取没拿到数值时，从原文兜底
_FACT_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def executor_node(state: AgentState) -> dict:
    """Build DAG from plan, execute ready tasks concurrently / 构建 DAG，并发执行就绪任务。

    Returns accumulated results for all tasks executed in this invocation.
    Tasks with no unmet dependencies run concurrently; downstream tasks whose
    predecessors failed are marked failed automatically.
    无未满足依赖的任务并发执行；上游失败的下游自动标记 failed。
    """
    plan = state.get("plan") or []
    if not plan:
        return {}

    # Build DAG index: task_id -> index
    task_ids = [_task_id(st, i) for i, st in enumerate(plan)]
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    # Determine which tasks are already done/failed
    completed: set[str] = set()
    failed: set[str] = set()
    existing_todos = state.get("todo_items") or []
    for t in existing_todos:
        tid = t.get("id", "") if isinstance(t, dict) else getattr(t, "id", "")
        status = t.get("status", "") if isinstance(t, dict) else getattr(t, "status", "")
        if status == "done":
            completed.add(tid)
        elif status == "failed":
            failed.add(tid)

    # Find runnable tasks: all deps completed, not yet completed/failed themselves
    runnable: list[tuple[int, object]] = []  # (index, sub_task)
    for i, st in enumerate(plan):
        tid = task_ids[i]
        if tid in completed or tid in failed:
            continue
        deps = getattr(st, "depends_on", []) or []
        unmet = [d for d in deps if d not in completed]
        if not unmet:
            runnable.append((i, st))
        # If any dep failed, mark this task as failed
        if any(d in failed for d in deps):
            failed.add(tid)

    if not runnable:
        # All tasks done or blocked — advance past any remaining
        next_cursor = len(plan)
        for i, st in enumerate(plan):
            if task_ids[i] not in completed and task_ids[i] not in failed:
                next_cursor = i
                break
        return {"plan_cursor": next_cursor}

    # Execute runnable tasks concurrently
    max_workers = min(len(runnable), 5)
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for idx, st in runnable:
            fut = pool.submit(_execute_one_task, st, idx, state)
            futures[fut] = (idx, st)

        for fut in as_completed(futures):
            idx, st = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                tid = task_ids[idx]
                logger.error(f"executor DAG: task {tid} crashed: {exc}")
                results.append(_task_failed(tid, idx, exc))

    # Merge results from all concurrent tasks
    merged = _merge_results(results, plan)

    # Compute plan_cursor as min(unfinished idx) over union of existing + this batch's completions.
    # 用所有已完成任务的并集计算 plan_cursor = min(未完成 idx)，避免 DAG 并行跳过中间节点。
    just_done: set[str] = set()
    just_failed: set[str] = set()
    for ti in merged.get("todo_items", []) or []:
        tid = ti.get("id", "") if isinstance(ti, dict) else getattr(ti, "id", "")
        status = ti.get("status", "") if isinstance(ti, dict) else getattr(ti, "status", "")
        if status == "done":
            just_done.add(tid)
        elif status == "failed":
            just_failed.add(tid)

    finished = completed | failed | just_done | just_failed
    next_cursor = len(plan)
    for i in range(len(plan)):
        if task_ids[i] not in finished:
            next_cursor = i
            break
    merged["plan_cursor"] = next_cursor
    return merged


def _task_id(sub_task, idx: int) -> str:
    """Get or generate a task_id for a SubTask."""
    tid = getattr(sub_task, "task_id", "")
    if tid:
        return tid
    # Generate a deterministic id from sub_query
    import hashlib
    h = hashlib.md5(sub_task.sub_query.encode()).hexdigest()[:8]
    return f"t-{idx}-{h}"


def _execute_one_task(sub_task, cursor: int, state: AgentState) -> dict:
    """Execute a single SubTask. Called from the DAG thread pool."""
    task_id = _task_id(sub_task, cursor)

    todo = TodoItem(
        id=task_id, sub_task_idx=cursor,
        title=getattr(sub_task, "sub_query", "")[:35],
        status="running", attempt=1, started_at=time.time(),
    )

    try:
        if sub_task.tool_calls:
            result = _run_tool_calls(sub_task, cursor, state)
        elif sub_task.expected_output_schema == "number":
            result = _run_calculation(sub_task, cursor, state)
        else:
            result = _run_retrieval(sub_task, cursor, state.get("doc_filter"), state)

        todo.status = "done"
        todo.finished_at = time.time()
        result["todo_items"] = [todo.model_dump()]
        result["todo_updates"] = [{"id": task_id, "status": "done", "finished_at": todo.finished_at}]
        return result

    except Exception as exc:
        err = classify_error(exc)
        err["node"] = "executor"
        err["timestamp"] = time.time()
        todo.status = "failed"
        todo.error = str(exc)
        todo.finished_at = time.time()
        logger.error(f"executor task {task_id} failed: {exc}")
        return {
            "error_log": [err],
            "todo_items": [todo.model_dump()],
            "todo_updates": [{"id": task_id, "status": "failed", "error": str(exc)}],
        }


def _task_failed(task_id: str, cursor: int, exc: Exception) -> dict:
    """Build a failed task result for a task that crashed outside _execute_one_task."""
    err = classify_error(exc)
    err["node"] = "executor"
    err["timestamp"] = time.time()
    todo = TodoItem(
        id=task_id, sub_task_idx=cursor,
        title="", status="failed", error=str(exc),
        finished_at=time.time(),
    )
    return {
        "error_log": [err],
        "todo_items": [todo.model_dump()],
        "todo_updates": [{"id": task_id, "status": "failed", "error": str(exc)}],
    }


def _merge_results(results: list[dict], plan: list) -> dict:
    """Merge multiple task results into a single state delta.
    Note: plan_cursor is intentionally NOT set here — it's computed by executor_node
    using the cumulative finished set, which is required to handle DAG parallelism
    correctly (max_idx+1 would skip pending intermediate tasks)."""
    merged: dict = {
        "retrieved_pages": [],
        "extracted_facts": [],
        "computed_values": [],
        "error_log": [],
        "todo_items": [],
        "todo_updates": [],
    }
    for r in results:
        for key in ("retrieved_pages", "extracted_facts", "computed_values",
                     "error_log", "todo_items", "todo_updates"):
            val = r.get(key, [])
            if val:
                merged[key].extend(val)
    return merged


def _run_tool_calls(sub_task, cursor: int, state: AgentState) -> dict:
    """Dispatch each ToolCall via the registry. Accumulate facts/pages/computed values.
    通过 Registry 分发每个 ToolCall，累积事实/页面/计算值。"""
    facts: list[Fact] = []
    pages: list = []
    cvs: list[ComputedValue] = []

    for tc in sub_task.tool_calls:
        tool_name = tc.get("tool", "") if isinstance(tc, dict) else getattr(tc, "tool", "")
        args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})

        if not tool_name:
            logger.warning("tool_calls entry missing 'tool' key; skipping")
            continue

        try:
            result = _registry_dispatch(tool_name, args)
        except KeyError as e:
            logger.warning(f"Unknown tool '{tool_name}' — falling back to legacy routing: {e}")
            return _run_retrieval(sub_task, cursor, state.get("doc_filter"), state)
        except Exception as e:
            logger.warning(f"Tool '{tool_name}' failed ({e}); skipping this tool call")
            continue

        # Collect results based on tool category
        spec = REGISTRY.get(tool_name)
        if spec and spec.category == "retrieval":
            result_pages = _extract_pages(result)
            pages.extend(result_pages)
            # Auto-read pages with VLM if tool was retrieve_pages
            if result_pages:
                with ThreadPoolExecutor(max_workers=min(len(result_pages), 5)) as ex:
                    texts = list(ex.map(
                        lambda h: vlm_read_page(
                            h.get("image_path", ""),
                            instruction=sub_task.sub_query,
                        ),
                        result_pages,
                    ))
                for hit, text in zip(result_pages, texts):
                    facts.append(Fact(
                        text=text,
                        source_doc=hit.get("doc_id", ""),
                        source_page=hit.get("page_num", 0),
                        sub_task_idx=cursor,
                    ))
        elif spec and spec.category == "compute":
            val = _extract_value(result)
            cvs.append(ComputedValue(expr=str(args), value=val, sub_task_idx=cursor))
        elif spec and spec.category == "reading":
            text = _extract_text(result)
            facts.append(Fact(
                text=text,
                source_doc=args.get("doc_id", "unknown"),
                source_page=args.get("page_num", 0),
                sub_task_idx=cursor,
            ))
        elif spec and spec.category == "resolution":
            # disambiguate_caliber: store results as facts
            auth = result.get("authoritative_fact", {}) if isinstance(result, dict) else {}
            explanation = result.get("explanation", "") if isinstance(result, dict) else ""
            facts.append(Fact(
                text=f"[口径消歧] {explanation}",
                source_doc=auth.get("doc_id", "unknown"),
                source_page=auth.get("page_num", 0),
                sub_task_idx=cursor,
            ))

    return {
        "retrieved_pages": pages,
        "extracted_facts": facts,
        "computed_values": cvs,
        "plan_cursor": cursor + 1,
    }


def _extract_pages(result) -> list[dict]:
    """Extract page hits from a tool result (dict, list, or pydantic model)."""
    if isinstance(result, dict):
        return result.get("pages") or []
    if isinstance(result, list):
        return result
    if hasattr(result, "pages"):
        return [p.model_dump() if hasattr(p, "model_dump") else p for p in result.pages]
    return []


def _extract_value(result) -> float:
    """Extract a float value from a calculator tool result."""
    if isinstance(result, dict):
        return float(result.get("value", float("nan")))
    if hasattr(result, "value"):
        return float(result.value)
    return float(result) if isinstance(result, (int, float)) else float("nan")


def _extract_text(result) -> str:
    """Extract text content from a VLM reading tool result."""
    if isinstance(result, dict):
        return result.get("extracted_text") or result.get("text") or str(result)
    if hasattr(result, "extracted_text"):
        return result.extracted_text
    return str(result)


def _run_calculation(sub_task, cursor: int, state: AgentState | None = None) -> dict:
    """Execute a 'number' schema sub-task via the AST-safe calculator.
    LLMCompiler-style placeholder resolution: `$<task_id>.value` is substituted
    with the structured numeric value from upstream tasks before calling the
    calculator. If any placeholder cannot be resolved, the calc step is SKIPPED
    (no NaN written) — the synthesizer can still proceed using the raw facts.
    占位符 $<task_id>.value 在调用计算器前用上游任务的结构化数值替换；
    解析失败则跳过本步（不写 NaN），synthesizer 仍可基于原始事实回答。"""
    expr = sub_task.sub_query or ""
    if state is not None and "$" in expr:
        resolved, missing = _resolve_placeholders(expr, state)
        if missing:
            logger.warning(
                f"calculator skipped — unresolved placeholders {missing} in '{expr}'"
            )
            return {}
        expr = resolved

    try:
        value = calculate(expr)
    except Exception as e:
        logger.warning(f"calculator rejected '{expr}': {e}")
        return {}
    cv = ComputedValue(expr=expr, value=value, sub_task_idx=cursor)
    return {"computed_values": [cv]}


def _resolve_placeholders(expr: str, state: AgentState) -> tuple[str, list[str]]:
    """Substitute `$<task_id>.value` tokens in `expr` with concrete numbers from state.
    用 state 中上游任务的具体数值替换 expr 里的 $<task_id>.value 占位符。

    Resolution order for each task_id:
      1. Find the SubTask whose `task_id` matches; let `idx` = its position in plan
      2. Prefer a `ComputedValue` at sub_task_idx==idx (chained calc support / 支持级联计算)
      3. Else prefer a `Fact` at sub_task_idx==idx whose structured `.value` is not None
      4. Else fall back to the first numeric literal parseable from `Fact.text`
      5. Otherwise the placeholder is unresolved.

    Returns (substituted_expr, missing_ids). If `missing_ids` is non-empty the
    caller MUST treat the calc as a no-op rather than running with placeholders.
    返回 (替换后表达式, 未解析 id 列表)；后者非空时调用方应跳过 calc。"""
    plan = state.get("plan") or []
    facts = state.get("extracted_facts") or []
    cvs = state.get("computed_values") or []

    # task_id → idx in plan / task_id 在 plan 中的位置
    id_to_idx: dict[str, int] = {}
    for i, st in enumerate(plan):
        tid = getattr(st, "task_id", "") if not isinstance(st, dict) else st.get("task_id", "")
        if tid:
            id_to_idx[tid] = i

    missing: list[str] = []
    out = expr

    for m in list(_PLACEHOLDER_RE.finditer(expr)):
        tid = m.group(1)
        idx = id_to_idx.get(tid)
        if idx is None:
            missing.append(tid)
            continue

        resolved_value: Optional[float] = None

        # 1. Chained calc — earlier ComputedValue at this idx
        for cv in cvs:
            cv_idx = getattr(cv, "sub_task_idx", None) if not isinstance(cv, dict) else cv.get("sub_task_idx")
            cv_val = getattr(cv, "value", None) if not isinstance(cv, dict) else cv.get("value")
            if cv_idx == idx and cv_val is not None and not _is_nan(cv_val):
                resolved_value = float(cv_val)
                break

        # 2. Structured fact value at this idx
        if resolved_value is None:
            for f in facts:
                f_idx = getattr(f, "sub_task_idx", None) if not isinstance(f, dict) else f.get("sub_task_idx")
                f_val = getattr(f, "value", None) if not isinstance(f, dict) else f.get("value")
                if f_idx == idx and f_val is not None:
                    resolved_value = float(f_val)
                    break

        # 3. Fallback — first numeric in fact.text at this idx
        if resolved_value is None:
            for f in facts:
                f_idx = getattr(f, "sub_task_idx", None) if not isinstance(f, dict) else f.get("sub_task_idx")
                if f_idx != idx:
                    continue
                text = getattr(f, "text", None) if not isinstance(f, dict) else f.get("text", "")
                if not text:
                    continue
                lit = _FACT_NUMERIC_RE.search(text)
                if lit:
                    try:
                        resolved_value = float(lit.group(0))
                        break
                    except ValueError:
                        pass

        if resolved_value is None:
            missing.append(tid)
            continue

        # Substitute every literal occurrence of this placeholder / 替换所有同名占位符
        out = out.replace(m.group(0), repr(resolved_value))

    return out, missing


def _is_nan(x) -> bool:
    """True if x is NaN (covers float('nan') and similar) / 判断是否为 NaN。"""
    try:
        return x != x  # NaN ≠ NaN
    except Exception:
        return False


def _run_retrieval(sub_task, cursor: int, session_doc_filter: list[str] | None, state: AgentState | None = None) -> dict:
    """Run ColQwen retrieval + concurrent VLM page reads for a text/table sub-task.
    执行 ColQwen 检索 + 并发 VLM 读页。Checks cross-turn cache before retrieval;
    applies remediation hints (widened top_k)."""
    if sub_task.target_doc:
        doc_filter = [sub_task.target_doc]
    elif session_doc_filter:
        doc_filter = session_doc_filter
    else:
        doc_filter = None

    # If remediation widened top_k, use it / 若修复策略放宽了 top_k 则使用
    hint = (state or {}).get("remediation_hint") or {}
    top_k = hint.get("widened_top_k", TOP_K)

    # Check known_facts before retrieval / 检索前先查跨轮缓存
    if state:
        known = state.get("known_facts") or []
        if known:
            cached_facts, cached_pages = _check_known_facts(sub_task.sub_query, known, cursor)
            if cached_facts:
                logger.info(f"executor: {len(cached_facts)} fact(s) served from cross-turn cache")
                return {
                    "retrieved_pages": cached_pages,
                    "extracted_facts": cached_facts,
                    "plan_cursor": cursor + 1,
                }

    hits = colpali_retrieve(sub_task.sub_query, top_k=top_k, doc_filter=doc_filter)

    if not hits:
        texts: list[str] = []
    else:
        with ThreadPoolExecutor(max_workers=min(len(hits), 5)) as ex:
            texts = list(ex.map(
                lambda h: vlm_read_page(h.image_path or "", instruction=sub_task.sub_query),
                hits,
            ))

    facts: list[Fact] = []
    for hit, text in zip(hits, texts):
        facts.append(Fact(
            text=text,
            source_doc=hit.doc_id,
            source_page=hit.page_num,
            sub_task_idx=cursor,
        ))

    # Structured fact extraction after VLM / VLM 后结构化抽取
    facts = _maybe_extract_structured(facts)

    return {
        "retrieved_pages": hits,
        "extracted_facts": facts,
        "plan_cursor": cursor + 1,
    }


# ---------------------------------------------------------------------------
# Structured fact extraction / 结构化事实抽取
# ---------------------------------------------------------------------------

def _maybe_extract_structured(facts: list[Fact]) -> list[Fact]:
    """Apply regex-based structured extraction to each Fact. Falls back to original on error.
    对每个 Fact 应用正则结构化抽取，异常时回退到原始 facts。"""
    try:
        from tools.fact_extractor import extract_structured_facts
        return extract_structured_facts(facts)
    except ImportError:
        return facts
    except Exception as e:
        logger.debug(f"structured fact extraction skipped: {e}")
        return facts


# ---------------------------------------------------------------------------
# Cross-turn fact cache lookup (semantic matching) / 跨轮事实缓存查找（语义匹配）
# ---------------------------------------------------------------------------

def _check_known_facts(sub_query: str, known_facts: list[dict], cursor: int) -> tuple[list[Fact], list]:
    """Check if known_facts can satisfy sub_query using semantic matching.
    用语义匹配检查已知事实是否能满足子查询。
    Hard hits (cosine > 0.85): skip retrieval entirely, serve from cache.
    Soft hits (cosine 0.5–0.85): return as retrieval priors.
    硬命中(>0.85)跳过检索直接返回；软命中(0.5-0.85)作为检索先验。"""
    from agent.memory import semantic_match
    from agent.state import PageHit

    result = semantic_match(sub_query, known_facts)
    hard = result["hard_hits"]
    soft = result["soft_hits"]

    if not hard:
        # Soft hits: return as priors but don't skip retrieval
        if soft:
            logger.info(f"executor: {len(soft)} soft-matched fact(s) as retrieval priors, score={result['best_score']}")
        return [], []

    matched_facts: list[Fact] = []
    matched_pages: list[PageHit] = []
    for kf in hard[:3]:  # cap at 3 hard hits
        matched_facts.append(Fact(
            text=kf.get("text", ""),
            source_doc=kf.get("source_doc", "cached(conv)"),
            source_page=kf.get("source_page", 0),
            sub_task_idx=cursor,
            entity=kf.get("entity", ""),
            period=kf.get("period", ""),
            metric=kf.get("metric", ""),
            value=kf.get("value"),
            unit=kf.get("unit"),
            raw_kind="numeric" if kf.get("value") is not None else "unstructured",
        ))
        matched_pages.append(PageHit(
            doc_id=kf.get("source_doc", "cached(conv)"),
            page_num=kf.get("source_page", 0),
            score=kf.get("_score", 0.85),
        ))

    logger.info(f"executor: {len(matched_facts)} fact(s) served from semantic cross-turn cache (score={result['best_score']})")
    return matched_facts, matched_pages
