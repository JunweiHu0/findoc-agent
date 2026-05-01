from __future__ import annotations

from loguru import logger

from tools.calculator import calculate
from tools.colpali_tool import colpali_retrieve
from tools.vlm_tool import vlm_read_page

from ..config import TOP_K
from ..state import AgentState, ComputedValue, Fact


def executor_node(state: AgentState) -> dict:
    cursor = state.get("plan_cursor", 0)
    plan = state.get("plan") or []
    if cursor >= len(plan):
        return {}

    sub_task = plan[cursor]
    if sub_task.expected_output_schema == "number":
        return _run_calculation(sub_task, cursor)
    return _run_retrieval(sub_task, cursor)


def _run_calculation(sub_task, cursor: int) -> dict:
    try:
        value = calculate(sub_task.sub_query)
    except Exception as e:
        logger.warning(f"calculator rejected '{sub_task.sub_query}': {e}")
        value = float("nan")
    cv = ComputedValue(expr=sub_task.sub_query, value=value, sub_task_idx=cursor)
    return {"computed_values": [cv], "plan_cursor": cursor + 1}


def _run_retrieval(sub_task, cursor: int) -> dict:
    doc_filter = [sub_task.target_doc] if sub_task.target_doc else None
    hits = colpali_retrieve(sub_task.sub_query, top_k=TOP_K, doc_filter=doc_filter)

    facts: list[Fact] = []
    for hit in hits:
        text = vlm_read_page(hit.image_path or "", instruction=sub_task.sub_query)
        facts.append(Fact(
            text=text,
            source_doc=hit.doc_id,
            source_page=hit.page_num,
            sub_task_idx=cursor,
        ))
    return {
        "retrieved_pages": hits,
        "extracted_facts": facts,
        "plan_cursor": cursor + 1,
    }
