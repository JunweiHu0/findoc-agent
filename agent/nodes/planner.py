from __future__ import annotations

import json

from loguru import logger

from ..config import INDEX_DIR
from ..llm import get_llm, has_llm_key
from ..prompts import load_prompt
from ..schemas import PlannerOutput
from ..state import AgentState, SubTask


_PROMPT = load_prompt("planner")


def _load_doc_metadata() -> str:
    mem_path = INDEX_DIR / "doc_memory.json"
    if not mem_path.exists():
        return "(no documents indexed yet)"
    try:
        data = json.loads(mem_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"failed to read doc_memory.json: {e}")
        return "(doc memory unreadable)"
    docs = data.get("docs") or []
    if not docs:
        return "(no documents indexed yet)"
    lines = [f"- doc_id={d['doc_id']}, pages={d['page_count']}" for d in docs]
    return "\n".join(lines)


def planner_node(state: AgentState) -> dict:
    if not has_llm_key():
        logger.warning("DEEPSEEK_API_KEY not set — planner falls back to single sub-task")
        return _fallback(state)

    try:
        llm = get_llm("planner").with_structured_output(PlannerOutput, method="json_mode")
        prompt = _PROMPT.format(
            query=state["query"],
            doc_metadata=_load_doc_metadata(),
        )
        result: PlannerOutput = llm.invoke(prompt)
        plan = [SubTask(**item.model_dump()) for item in result.plan]
        if not plan:
            return _fallback(state)
        return {"plan": plan, "plan_cursor": 0}
    except Exception as e:
        logger.warning(f"planner LLM call failed ({e}); falling back to single sub-task")
        return _fallback(state)


def _fallback(state: AgentState) -> dict:
    return {"plan": [SubTask(sub_query=state["query"])], "plan_cursor": 0}
