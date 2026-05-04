"""Verifier node — judges evidence sufficiency + consistency (P19: structured root-cause output).

Evaluates whether the collected facts and computed values are sufficient to answer
the user's query. When insufficient, produces structured MissingFact entries with
root_cause classification to drive downstream remediation (P20).
"""

from __future__ import annotations

from loguru import logger

from ..llm import get_llm, has_llm_key
from ..prompts import load_prompt
from ..schemas import VerifierOutput
from ..state import AgentState, SubTask


_PROMPT = load_prompt("verifier")


def verifier_node(state: AgentState) -> dict:
    """Judge evidence sufficiency and cross-page consistency.

    Returns a state delta with is_sufficient, confidence, and optional
    structured missing_facts for the remediation node to consume.
    """
    iter_count = state.get("reflexion_iter", 0) + 1
    plan = state.get("plan") or []
    cursor = state.get("plan_cursor", 0)

    if not has_llm_key():
        logger.warning("DEEPSEEK_API_KEY not set — verifier uses plan-exhausted heuristic")
        return _heuristic(plan, cursor, iter_count)

    # P19: structured missing_facts
    try:
        llm = get_llm("verifier").with_structured_output(VerifierOutput, method="json_mode")
        prompt = _PROMPT.format(
            query=state["query"],
            plan=_render_plan(plan),
            evidence=_render_evidence(state),
            tried_queries=_render_tried(state, "tried_queries"),
            tried_pages=_render_tried(state, "tried_pages"),
        )
        result: VerifierOutput = llm.invoke(prompt)
    except Exception as e:
        logger.warning(f"verifier LLM call failed ({e}); using plan-exhausted heuristic")
        return _heuristic(plan, cursor, iter_count)

    # Collect tried markers from this cycle
    tried_queries = _collect_tried_queries(plan, cursor)
    tried_pages = _collect_tried_pages(state)

    # P19: early-stop — no new facts gained in this reflexion round
    prev_fact_count = _count_facts_before_cursor(state, cursor)
    if iter_count > 1 and prev_fact_count == len(state.get("extracted_facts") or []):
        logger.info("reflexion early-stop: no new facts gained this round; forcing synthesis")
        return {
            "reflexion_iter": iter_count,
            "is_sufficient": True,
            "confidence": 0.3,
            "missing_facts": [],
            "tried_queries": tried_queries,
        }

    # Build missing_facts dicts for state accumulation
    missing_dicts = [mf.model_dump() for mf in (result.missing_facts or [])]

    if result.is_sufficient and not result.inconsistency:
        return {
            "reflexion_iter": iter_count,
            "is_sufficient": True,
            "confidence": result.confidence,
            "missing_facts": [],
            "tried_queries": tried_queries,
        }

    # Not sufficient — build follow-up SubTasks from structured missing_facts
    new_subtasks = _missing_facts_to_subtasks(result.missing_facts)
    if not new_subtasks:
        # Fallback: use old missing_info string or inconsistency text
        fallback = result.inconsistency or getattr(result, "missing_info", "") or "需要更多证据"
        new_subtasks = [SubTask(sub_query=fallback)]

    new_plan = list(plan) + new_subtasks
    return {
        "plan": new_plan,
        "reflexion_iter": iter_count,
        "is_sufficient": False,
        "confidence": result.confidence,
        "missing_facts": missing_dicts,
        "tried_queries": tried_queries,
        "tried_pages": tried_pages,
    }


def _missing_facts_to_subtasks(missing_facts: list) -> list[SubTask]:
    """Convert structured MissingFact list into SubTasks for the executor.

    At P19, all root_causes map to a retrieval+VLM SubTask (the basic path).
    P20 remediation_node later applies differentiated strategies.
    """
    out: list[SubTask] = []
    for mf in (missing_facts or []):
        query = mf.suggested_query or mf.what
        target = mf.suggested_target_doc
        out.append(SubTask(
            sub_query=query,
            target_doc=target,
            expected_output_schema="text",
        ))
    return out


def _count_facts_before_cursor(state: AgentState, cursor: int) -> int:
    """Count extracted_facts whose sub_task_idx < cursor (facts from prior rounds)."""
    facts = state.get("extracted_facts") or []
    return sum(1 for f in facts if (f.sub_task_idx or 0) < cursor)


def _collect_tried_queries(plan: list[SubTask], cursor: int) -> list[str]:
    """Collect sub_queries already executed (up to cursor)."""
    return [p.sub_query for p in (plan or [])[:cursor]]


def _collect_tried_pages(state: AgentState) -> list[dict]:
    """Collect unique (doc_id, page_num) pairs already retrieved."""
    pages = state.get("retrieved_pages") or []
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for p in pages:
        key = (p.doc_id, p.page_num)
        if key not in seen:
            seen.add(key)
            out.append({"doc_id": p.doc_id, "page_num": p.page_num})
    return out


def _render_tried(state: AgentState, key: str) -> str:
    """Render tried_queries or tried_pages as bullet list for the verifier prompt."""
    items = state.get(key) or []
    if not items:
        return "(none yet)"
    if key == "tried_queries":
        return "\n".join(f"- {q}" for q in items[-20:])
    # tried_pages
    return "\n".join(f"- {p.get('doc_id','?')} p.{p.get('page_num','?')}" for p in items[-30:])


def _heuristic(plan: list[SubTask], cursor: int, iter_count: int) -> dict:
    """Fallback sufficiency check: sufficient iff all plan steps have been executed."""
    sufficient = cursor >= len(plan)
    return {
        "reflexion_iter": iter_count,
        "is_sufficient": sufficient,
        "confidence": 0.5 if sufficient else 0.3,
        "missing_facts": [],
        "missing_info": "" if sufficient else "plan not yet exhausted",
    }


def _render_plan(plan: list[SubTask]) -> str:
    """Format the execution plan as numbered lines for the verifier prompt."""
    if not plan:
        return "(empty)"
    return "\n".join(
        f"{i}. {p.sub_query} (target={p.target_doc}, schema={p.expected_output_schema})"
        for i, p in enumerate(plan)
    )


def _render_evidence(state: AgentState) -> str:
    """Format extracted facts and computed values for the verifier prompt."""
    facts = state.get("extracted_facts") or []
    cvs = state.get("computed_values") or []
    if not facts and not cvs:
        return "(no evidence yet)"
    lines = []
    for f in facts:
        # P24: include structured keys when available
        extra = ""
        if f.entity or f.metric:
            extra_parts = []
            if f.entity: extra_parts.append(f.entity)
            if f.period: extra_parts.append(f.period)
            if f.metric: extra_parts.append(f.metric)
            if f.value is not None: extra_parts.append(str(f.value))
            if f.unit: extra_parts.append(f.unit)
            extra = f"  [{', '.join(extra_parts)}]"
        lines.append(f"- [{f.source_doc} p.{f.source_page}]{extra} {f.text}")
    for c in cvs:
        lines.append(f"- [calc] {c.expr} = {c.value}")
    return "\n".join(lines)
