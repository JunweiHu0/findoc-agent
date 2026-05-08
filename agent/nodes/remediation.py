"""Remediation node — applies differentiated fixes per root_cause with budget control / 差异化修复节点——按根因分派修复策略，含预算控制。"""

from __future__ import annotations

import time

from loguru import logger

from ..state import AgentState, SubTask, TodoItem


# Budget defaults
_DEFAULT_RETRIEVAL_BUDGET = 10
_DEFAULT_VLM_BUDGET = 20


def remediation_node(state: AgentState) -> dict:
    """Apply differentiated fixes based on missing_facts[].root_cause / 根据缺失事实的根因分派差异化修复

    Returns state delta that the executor will consume on the next pass / 返回状态增量供执行器下一轮消费
    """
    missing = state.get("missing_facts") or []
    plan = list(state.get("plan") or [])
    budget_r = state.get("budget_retrievals", _DEFAULT_RETRIEVAL_BUDGET)
    budget_v = state.get("budget_vlm_calls", _DEFAULT_VLM_BUDGET)

    if not missing:
        logger.info("remediation: no missing_facts, forcing synthesis fallthrough")
        return {"is_sufficient": True}

    new_subtasks: list[SubTask] = []
    new_todos: list[dict] = []
    todo_updates: list[dict] = []

    existing_todos = state.get("todo_items") or []

    for mf in missing:
        if budget_r <= 0 and budget_v <= 0:
            logger.warning("remediation: budget exhausted, stopping")
            break

        root = mf.get("root_cause", "retrieval_miss")
        sub_task_idx = mf.get("sub_task_idx", -1)
        logger.info(f"remediation: processing root_cause={root}, what={mf.get('what','?')}")

        # find original todo to use as parent / 查找原始 todo 作为父节点
        parent_id = ""
        attempt = 1
        if sub_task_idx >= 0:
            for t in existing_todos:
                t_idx = t.get("sub_task_idx", -1) if isinstance(t, dict) else getattr(t, "sub_task_idx", -1)
                if t_idx == sub_task_idx:
                    parent_id = t.get("id", "") if isinstance(t, dict) else getattr(t, "id", "")
                    prev_attempt = t.get("attempt", 0) if isinstance(t, dict) else getattr(t, "attempt", 0)
                    attempt = prev_attempt + 1
                    break

        st = None
        if root == "retrieval_miss":
            if budget_r <= 0:
                continue
            st = _remediate_retrieval_miss(mf)
            budget_r -= 1
            budget_v -= 1

        elif root == "reading_miss":
            if budget_v <= 0:
                continue
            st = _remediate_reading_miss(mf, state)
            budget_v -= 1

        elif root == "ambiguous_query":
            if budget_r <= 0:
                continue
            st = _remediate_ambiguous_query(mf)
            budget_r -= 1
            budget_v -= 1

        elif root == "inconsistency":
            if budget_v <= 0:
                continue
            st = _remediate_inconsistency(mf, state)
            budget_v -= 2

        if st is not None:
            new_subtasks.append(st)
            # create retry todo item with incremented attempt / 创建重试 todo 项，attempt 递增
            new_idx = len(plan) + len(new_subtasks) - 1
            todo = TodoItem(
                id=f"t-retry-{new_idx}",
                sub_task_idx=new_idx,
                title=st.sub_query[:35],
                status="pending",
                attempt=attempt,
                parent_id=parent_id,
                started_at=time.time(),
            )
            new_todos.append(todo.model_dump())
            todo_updates.append({
                "id": todo.id,
                "status": "pending",
                "attempt": attempt,
                "parent_id": parent_id,
            })

    # Always write back current budget snapshots — single source of truth / 始终写回当前预算快照
    budget_deltas = {"budget_retrievals": budget_r, "budget_vlm_calls": budget_v}

    if not new_subtasks:
        logger.info("remediation: no actionable fixes (budget exhausted or all root_causes handled)")
        return {**budget_deltas, "is_sufficient": True}

    return {
        "plan": plan + new_subtasks,
        "todo_items": new_todos,
        "todo_updates": todo_updates,
        **budget_deltas,
    }


# ---------------------------------------------------------------------------
# Per-root-cause SubTask builders
# ---------------------------------------------------------------------------

def _remediate_retrieval_miss(mf: dict) -> SubTask:
    """Build a SubTask for broader re-retrieval with rewritten query and optional doc constraint."""
    query = mf.get("suggested_query") or mf.get("what", "")
    target = mf.get("suggested_target_doc")
    return SubTask(
        sub_query=query,
        target_doc=target,
        expected_output_schema="text",
    )


def _remediate_reading_miss(mf: dict, state: AgentState) -> SubTask:
    """Build a SubTask that re-reads the SAME pages via explicit read_page_with_vlm tool_calls — no re-retrieval.
    构建带显式 tool_calls 的 SubTask，从 state.retrieved_pages 查 image_path，跳过检索直接重读相同页面。"""
    query = mf.get("suggested_query") or mf.get("what", "")
    page_nums = mf.get("suggested_page_nums") or []
    target = mf.get("suggested_target_doc")

    # Look up image_paths for the requested (target_doc, page_num) pairs from state.retrieved_pages
    # 从 state.retrieved_pages 反查所需页面的 image_path
    retrieved = state.get("retrieved_pages") or []
    page_index: dict[tuple[str, int], str] = {}
    for p in retrieved:
        doc_id = getattr(p, "doc_id", None) or (p.get("doc_id") if isinstance(p, dict) else None)
        page_num = getattr(p, "page_num", None) or (p.get("page_num") if isinstance(p, dict) else None)
        image_path = getattr(p, "image_path", None) or (p.get("image_path") if isinstance(p, dict) else None)
        if doc_id and page_num is not None and image_path:
            page_index[(doc_id, int(page_num))] = image_path

    tool_calls: list[dict] = []
    for pn in page_nums:
        if target and (target, int(pn)) in page_index:
            tool_calls.append({
                "tool": "read_page_with_vlm",
                "args": {"image_path": page_index[(target, int(pn))], "instruction": query},
            })
        else:
            # Fall back: any retrieved page matching the page_num
            for (doc_id, p_num), img in page_index.items():
                if p_num == int(pn):
                    tool_calls.append({
                        "tool": "read_page_with_vlm",
                        "args": {"image_path": img, "instruction": query},
                    })
                    break

    if tool_calls:
        return SubTask(
            sub_query=query,
            target_doc=target,
            expected_output_schema="text",
            tool_calls=tool_calls,
        )

    # Fallback: no image_paths found, degrade gracefully to re-retrieval with a clean query.
    # Don't pollute the query with prefix tags — just retry the natural query.
    # 找不到原始 image_path 时回退到检索，但不污染 query 字面量
    return SubTask(
        sub_query=query,
        target_doc=target,
        expected_output_schema="text",
    )


def _remediate_ambiguous_query(mf: dict) -> SubTask:
    """Build a SubTask with a rewritten, fully self-contained sub_query."""
    query = mf.get("suggested_query") or mf.get("what", "")
    target = mf.get("suggested_target_doc")
    return SubTask(
        sub_query=query,
        target_doc=target,
        expected_output_schema="text",
    )


def _remediate_inconsistency(mf: dict, state: AgentState) -> SubTask:
    """Build a SubTask that explicitly invokes the disambiguate_caliber tool on conflicting facts.
    构建显式调用 disambiguate_caliber 工具的 SubTask，用 state.extracted_facts 中相关事实作为冲突输入。"""
    query = mf.get("suggested_query") or mf.get("what", "")
    target = mf.get("suggested_target_doc")
    sub_task_idx = mf.get("sub_task_idx", -1)

    # Gather candidate conflicting facts: prefer those linked to the same sub_task_idx,
    # otherwise the most recent extracted_facts. 收集冲突候选事实
    facts = state.get("extracted_facts") or []
    fact_texts: list[str] = []
    if sub_task_idx >= 0:
        for f in facts:
            f_idx = getattr(f, "sub_task_idx", None) if not isinstance(f, dict) else f.get("sub_task_idx")
            if f_idx == sub_task_idx:
                text = getattr(f, "text", None) if not isinstance(f, dict) else f.get("text", "")
                if text:
                    fact_texts.append(text)
    if not fact_texts:
        # Fallback: take last 4 facts
        for f in facts[-4:]:
            text = getattr(f, "text", None) if not isinstance(f, dict) else f.get("text", "")
            if text:
                fact_texts.append(text)

    if len(fact_texts) >= 2:
        return SubTask(
            sub_query=query,
            target_doc=target,
            expected_output_schema="text",
            tool_calls=[{
                "tool": "disambiguate_caliber",
                "args": {"conflict_topic": query, "fact_texts": fact_texts[:6]},
            }],
        )

    # Fallback: not enough facts to disambiguate; just retry retrieval with the clean query.
    # 候选事实不足时回退到普通检索，不再污染 sub_query 字面量
    return SubTask(
        sub_query=query,
        target_doc=target,
        expected_output_schema="text",
    )
