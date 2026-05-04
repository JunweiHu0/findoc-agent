"""Remediation node (P20) — pre-processes state before executor re-entry.

Reads structured missing_facts from the verifier and applies the
appropriate fix per root_cause, rather than blindly appending a
retrieval+VLM SubTask like the old code did.
"""

from __future__ import annotations

from loguru import logger

from ..state import AgentState, SubTask


# Budget defaults
_DEFAULT_RETRIEVAL_BUDGET = 10
_DEFAULT_VLM_BUDGET = 20


def remediation_node(state: AgentState) -> dict:
    """Apply differentiated fixes based on missing_facts[].root_cause.

    Returns state delta that the executor will consume on the next pass.
    Four fix strategies: retrieval_miss (re-retrieve), reading_miss (re-read
    same pages with refined instruction), ambiguous_query (rewrite query),
    inconsistency (trigger caliber disambiguation).
    """
    missing = state.get("missing_facts") or []
    plan = list(state.get("plan") or [])
    budget_r = state.get("budget_retrievals", _DEFAULT_RETRIEVAL_BUDGET)
    budget_v = state.get("budget_vlm_calls", _DEFAULT_VLM_BUDGET)

    if not missing:
        # No structured missing facts — nothing to remediate; force fallthrough
        logger.info("remediation: no missing_facts, forcing synthesis fallthrough")
        return {"is_sufficient": True, "confidence": 0.3}

    new_subtasks: list[SubTask] = []
    budget_deltas: dict = {}

    for mf in missing:
        # Budget check before processing each missing fact
        if budget_r <= 0 and budget_v <= 0:
            logger.warning("remediation: budget exhausted, stopping")
            break

        root = mf.get("root_cause", "retrieval_miss")
        logger.info(f"remediation: processing root_cause={root}, what={mf.get('what','?')}")

        if root == "retrieval_miss":
            if budget_r <= 0:
                continue
            st = _remediate_retrieval_miss(mf)
            budget_r -= 1
            budget_v -= 1  # retrieval implies at least one VLM read
            budget_deltas = {"budget_retrievals": budget_r, "budget_vlm_calls": budget_v}
            new_subtasks.append(st)

        elif root == "reading_miss":
            if budget_v <= 0:
                continue
            st = _remediate_reading_miss(mf)
            budget_v -= 1
            budget_deltas = {"budget_vlm_calls": budget_v}
            new_subtasks.append(st)

        elif root == "ambiguous_query":
            if budget_r <= 0:
                continue
            st = _remediate_ambiguous_query(mf)
            budget_r -= 1
            budget_v -= 1
            budget_deltas = {"budget_retrievals": budget_r, "budget_vlm_calls": budget_v}
            new_subtasks.append(st)

        elif root == "inconsistency":
            if budget_v <= 0:
                continue
            st = _remediate_inconsistency(mf)
            budget_v -= 2  # inconsistency costs more (multi-page VLM)
            budget_deltas = {"budget_vlm_calls": budget_v}
            new_subtasks.append(st)

    if not new_subtasks:
        logger.info("remediation: no actionable fixes (budget exhausted or all root_causes handled)")
        return {**budget_deltas, "is_sufficient": True, "confidence": 0.2}

    # Append new subtasks to plan; executor will process them
    result: dict = {
        "plan": plan + new_subtasks,
        **budget_deltas,
    }
    return result


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


def _remediate_reading_miss(mf: dict) -> SubTask:
    """Build a SubTask to re-read the SAME pages with a refined VLM instruction — no re-retrieval."""
    query = mf.get("suggested_query") or mf.get("what", "")
    page_nums = mf.get("suggested_page_nums") or []
    target = mf.get("suggested_target_doc")

    # Build a precise re-read instruction
    if page_nums:
        hint = f"（仅重读第 {', '.join(str(p) for p in page_nums)} 页）"
    else:
        hint = "（重读之前检索到的页面）"
    instruction = f"[重读]{hint} {query}"

    return SubTask(
        sub_query=instruction,
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


def _remediate_inconsistency(mf: dict) -> SubTask:
    """Build a SubTask to trigger caliber disambiguation on conflicting pages."""
    query = mf.get("suggested_query") or mf.get("what", "")
    target = mf.get("suggested_target_doc")
    return SubTask(
        sub_query=f"[口径消歧] {query}",
        target_doc=target,
        expected_output_schema="text",
    )
