from __future__ import annotations

from loguru import logger

from ..llm import get_llm, has_llm_key
from ..prompts import load_prompt
from ..schemas import PlannerOutput
from ..state import AgentState, SubTask


_PROMPT = load_prompt("planner")


def planner_node(state: AgentState) -> dict:
    if not has_llm_key():
        logger.warning("DEEPSEEK_API_KEY not set — planner falls back to single sub-task")
        return _fallback(state)

    try:
        llm = get_llm("planner").with_structured_output(PlannerOutput)
        prompt = _PROMPT.format(
            query=state["query"],
            doc_metadata="(no document memory yet — P2 will provide this)",
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
