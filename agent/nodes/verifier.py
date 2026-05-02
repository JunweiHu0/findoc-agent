from __future__ import annotations

from loguru import logger

from ..llm import get_llm, has_llm_key
from ..prompts import load_prompt
from ..schemas import VerifierOutput
from ..state import AgentState, SubTask


_PROMPT = load_prompt("verifier")


def verifier_node(state: AgentState) -> dict:
    iter_count = state.get("reflexion_iter", 0) + 1
    plan = state.get("plan") or []
    cursor = state.get("plan_cursor", 0)

    if not has_llm_key():
        logger.warning("DEEPSEEK_API_KEY not set — verifier uses plan-exhausted heuristic")
        return _heuristic(plan, cursor, iter_count)

    try:
        llm = get_llm("verifier").with_structured_output(VerifierOutput, method="json_mode")
        prompt = _PROMPT.format(
            query=state["query"],
            plan=_render_plan(plan),
            evidence=_render_evidence(state),
        )
        result: VerifierOutput = llm.invoke(prompt)
    except Exception as e:
        logger.warning(f"verifier LLM call failed ({e}); using plan-exhausted heuristic")
        return _heuristic(plan, cursor, iter_count)

    if result.is_sufficient and not result.inconsistency:
        return {
            "reflexion_iter": iter_count,
            "is_sufficient": True,
            "missing_info": "",
        }

    follow_up = result.missing_info or result.inconsistency or "需要更多证据"
    new_plan = list(plan) + [SubTask(sub_query=follow_up)]
    return {
        "plan": new_plan,
        "reflexion_iter": iter_count,
        "is_sufficient": False,
        "missing_info": follow_up,
    }


def _heuristic(plan: list[SubTask], cursor: int, iter_count: int) -> dict:
    sufficient = cursor >= len(plan)
    return {
        "reflexion_iter": iter_count,
        "is_sufficient": sufficient,
        "missing_info": "" if sufficient else "plan not yet exhausted",
    }


def _render_plan(plan: list[SubTask]) -> str:
    if not plan:
        return "(empty)"
    return "\n".join(f"{i}. {p.sub_query} (target={p.target_doc}, schema={p.expected_output_schema})" for i, p in enumerate(plan))


def _render_evidence(state: AgentState) -> str:
    facts = state.get("extracted_facts") or []
    cvs = state.get("computed_values") or []
    if not facts and not cvs:
        return "(no evidence yet)"
    lines = []
    for f in facts:
        lines.append(f"- [{f.source_doc} p.{f.source_page}] {f.text}")
    for c in cvs:
        lines.append(f"- [calc] {c.expr} = {c.value}")
    return "\n".join(lines)
