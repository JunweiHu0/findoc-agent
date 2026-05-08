"""Plan Critic node — on-demand plan revision triggered by signal words or task failures / 计划评审节点——由信号词或任务失败触发的按需计划修订。"""

from __future__ import annotations

from loguru import logger

from ..llm import get_llm, has_llm_key
from ..state import AgentState, SubTask


# Signal words that indicate a plan may need revision
_SIGNAL_WORDS = [
    "口径变更", "合并范围调整", "重述会计政策", "追溯调整",
    "会计估计变更", "前期差错更正", "分部报告", "终止经营",
]

# Hard cap on plan_critic invocations to prevent runaway oscillation / plan_critic 调用硬上限，防止振荡
_MAX_CRITIC_ITER = 2


def _has_signal(facts: list) -> bool:
    """Check if any fact text contains plan-revision signal words / 检查事实文本是否含计划修订信号词。"""
    for f in facts:
        text = getattr(f, "text", "") or ""
        for sw in _SIGNAL_WORDS:
            if sw in text:
                return True
    return False


def _has_failed(state: AgentState) -> bool:
    """Check if any todo item has status=failed / 检查是否有 todo 项状态为 failed。"""
    todos = state.get("todo_items") or []
    for t in todos:
        status = t.get("status", "") if isinstance(t, dict) else getattr(t, "status", "")
        if status == "failed":
            return True
    return False


def _should_trigger(state: AgentState) -> bool:
    """Determine whether plan_critic should activate / 判断是否应激活计划评审。

    Guards against re-entry: plan_critic only fires when (1) signal/failure exists AND
    (2) plan_cursor has advanced since last critique AND (3) iter cap not reached.
    Both signals (extracted_facts and failed todos) are sticky — they never clear —
    so without these guards executor → plan_critic → executor would oscillate.
    重入保护：仅在游标推进且未达上限时触发，否则信号永远存在会导致振荡。"""
    iter_count = state.get("plan_critic_iter", 0)
    if iter_count >= _MAX_CRITIC_ITER:
        return False

    cursor = state.get("plan_cursor", 0)
    last_cursor = state.get("plan_critic_last_cursor", -1)
    if cursor <= last_cursor:
        return False

    facts = state.get("extracted_facts") or []
    return _has_signal(facts) or _has_failed(state)


def plan_critic_node(state: AgentState) -> dict:
    """Evaluate remaining plan steps and suggest revisions when signal words or failures are detected / 检测到信号词或失败时评估剩余计划步骤，建议修订。"""
    if not _should_trigger(state):
        return {}

    # Always stamp guard fields so subsequent _should_trigger sees this run / 写入幂等保护字段
    guard = {
        "plan_critic_last_cursor": state.get("plan_cursor", 0),
        "plan_critic_iter": state.get("plan_critic_iter", 0) + 1,
    }

    plan = list(state.get("plan") or [])
    todos = state.get("todo_items") or []
    completed_ids = {
        t.get("id", "") if isinstance(t, dict) else getattr(t, "id", "")
        for t in todos
        if (t.get("status", "") if isinstance(t, dict) else getattr(t, "status", "")) == "done"
    }
    failed_ids = {
        t.get("id", "") if isinstance(t, dict) else getattr(t, "id", "")
        for t in todos
        if (t.get("status", "") if isinstance(t, dict) else getattr(t, "status", "")) == "failed"
    }

    # Build remaining plan description
    remaining = [st for st in plan if getattr(st, "task_id", "") not in completed_ids]
    if not remaining:
        return guard

    remaining_desc = "\n".join(
        f"- {getattr(st, 'task_id', '?')}: {st.sub_query}"
        for st in remaining
    )

    if not has_llm_key():
        logger.info("plan_critic: no LLM key, skipping revision")
        return guard

    try:
        llm = get_llm("planner")
        prompt = (
            "你是计划评审员。当前Agent执行计划中出现了异常信号（口径变更、任务失败等）。"
            "请评估剩余计划是否仍然合理。\n\n"
            f"已完成任务: {len(completed_ids)} 个\n"
            f"失败任务: {len(failed_ids)} 个\n"
            f"剩余计划:\n{remaining_desc}\n\n"
            "返回JSON，只含一个字段:\n"
            '{"revise": true/false, "reason": "说明", '
            '"new_subtasks": [...], "drop_task_ids": [...]}\n\n'
            "如果原计划仍然合理，revise=false。"
            "如果需要修改，在new_subtasks中列出新的子任务（每个有sub_query, target_doc可选），"
            "在drop_task_ids中列出应跳过的失败任务ID。"
        )
        result = llm.invoke(prompt)
        content = result.content.strip()

        # Extract JSON from LLM response
        import json
        try:
            # Find JSON block
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
            else:
                parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"plan_critic: failed to parse LLM response: {content[:200]}")
            return guard

        if not parsed.get("revise", False):
            return guard

        # Build new SubTasks from the revision
        new_sts = []
        for ns in parsed.get("new_subtasks", []):
            new_sts.append(SubTask(
                sub_query=ns.get("sub_query", ""),
                target_doc=ns.get("target_doc"),
                expected_output_schema=ns.get("expected_output_schema", "text"),
            ))

        drop_ids = parsed.get("drop_task_ids", [])
        # Mark dropped tasks in todos
        todo_updates = [{"id": did, "status": "skipped"} for did in drop_ids]

        logger.info(f"plan_critic: revising plan — adding {len(new_sts)} tasks, dropping {len(drop_ids)}")
        return {
            **guard,
            "plan": plan + new_sts,
            "todo_updates": todo_updates,
        }

    except Exception as e:
        logger.warning(f"plan_critic: LLM call failed ({e}), keeping original plan")
        return guard


def should_trigger_plan_critic(state: AgentState) -> bool:
    """Condition function for graph edge routing / 图边路由的条件函数。"""
    return _should_trigger(state)
