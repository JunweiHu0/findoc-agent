"""Executor node — dispatches sub-tasks via tool registry (P21) or legacy schema routing.

P29 DAG scheduling: builds a dependency graph from plan SubTasks, runs
independent tasks at the same topological level concurrently via ThreadPoolExecutor
(max 5 workers). Single task failure does not crash the whole graph — downstream
dependents are marked failed and the verifier handles recovery.

VLM reads are run concurrently per task (max 5 workers).
P25 cross-turn fact cache is checked before any retrieval.
P24 structured fact extraction runs after every VLM read.
P26: tool failures write error_log entries, never silent.
P26.5: todo_items track runtime status per task.
"""

from __future__ import annotations

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


def executor_node(state: AgentState) -> dict:
    """P29: Build DAG from plan, execute all ready tasks concurrently by topological level.

    Returns accumulated results for all tasks executed in this invocation.
    Tasks with no unmet dependencies run concurrently; downstream tasks whose
    predecessors failed are marked failed automatically.
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
    return _merge_results(results, plan)


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
            result = _run_calculation(sub_task, cursor)
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
    """Merge multiple task results into a single state delta."""
    merged: dict = {
        "retrieved_pages": [],
        "extracted_facts": [],
        "computed_values": [],
        "error_log": [],
        "todo_items": [],
        "todo_updates": [],
    }

    max_cursor = 0
    for r in results:
        for key in ("retrieved_pages", "extracted_facts", "computed_values",
                     "error_log", "todo_items", "todo_updates"):
            val = r.get(key, [])
            if val:
                merged[key].extend(val)
        # Track highest cursor seen
        for ti in r.get("todo_items", []):
            idx = ti.get("sub_task_idx", 0) if isinstance(ti, dict) else getattr(ti, "sub_task_idx", 0)
            if idx is not None and idx + 1 > max_cursor:
                max_cursor = idx + 1

    # If nothing executed, advance to end
    if not results:
        max_cursor = len(plan)

    merged["plan_cursor"] = max_cursor
    return merged


def _run_tool_calls(sub_task, cursor: int, state: AgentState) -> dict:
    """P21: Dispatch each ToolCall via the registry. Accumulate facts/pages/computed values."""
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


def _run_calculation(sub_task, cursor: int) -> dict:
    """Execute a 'number' schema sub-task via the AST-safe calculator."""
    try:
        value = calculate(sub_task.sub_query)
    except Exception as e:
        logger.warning(f"calculator rejected '{sub_task.sub_query}': {e}")
        value = float("nan")
    cv = ComputedValue(expr=sub_task.sub_query, value=value, sub_task_idx=cursor)
    return {"computed_values": [cv], "plan_cursor": cursor + 1}


def _run_retrieval(sub_task, cursor: int, session_doc_filter: list[str] | None, state: AgentState | None = None) -> dict:
    """Run ColQwen retrieval + concurrent VLM page reads for a text/table sub-task.

    Checks cross-turn fact cache (P25) before retrieval. Applies P20 remediation
    hints (widened top_k). Runs VLM on retrieved pages concurrently (max 5 workers).
    """
    if sub_task.target_doc:
        doc_filter = [sub_task.target_doc]
    elif session_doc_filter:
        doc_filter = session_doc_filter
    else:
        doc_filter = None

    # P20: if remediation widened top_k, use it
    hint = (state or {}).get("remediation_hint") or {}
    top_k = hint.get("widened_top_k", TOP_K)

    # P25: check known_facts before retrieval
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

    # P24: run structured fact extraction after VLM
    facts = _maybe_extract_structured(facts)

    return {
        "retrieved_pages": hits,
        "extracted_facts": facts,
        "plan_cursor": cursor + 1,
    }


# ---------------------------------------------------------------------------
# P24: Structured fact extraction
# ---------------------------------------------------------------------------

def _maybe_extract_structured(facts: list[Fact]) -> list[Fact]:
    """Apply regex-based structured extraction to each Fact (P24). Falls back to original facts on error."""
    try:
        from tools.fact_extractor import extract_structured_facts
        return extract_structured_facts(facts)
    except ImportError:
        return facts
    except Exception as e:
        logger.debug(f"structured fact extraction skipped: {e}")
        return facts


# ---------------------------------------------------------------------------
# P28: Cross-turn fact cache lookup (semantic matching)
# ---------------------------------------------------------------------------

def _check_known_facts(sub_query: str, known_facts: list[dict], cursor: int) -> tuple[list[Fact], list]:
    """Check if known_facts can satisfy this sub_query using semantic matching (P28).

    Hard hits (cosine > 0.85): skip retrieval entirely, serve from cache.
    Soft hits (cosine 0.5–0.85): return as retrieval_priors for planner weighting.
    """
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
