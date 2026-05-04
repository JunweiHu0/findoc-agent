"""Planner node — decomposes user question into ordered sub-tasks.

P22: retrieval_scout_node provides pre-retrieval context so the planner
can make informed target_doc decisions instead of blind guessing.
"""

from __future__ import annotations

import json

from loguru import logger

from ..config import INDEX_DIR
from ..llm import get_llm, has_llm_key
from ..prompts import load_prompt
from ..schemas import PlannerOutput
from ..state import AgentState, SubTask
from tools.registry import get_tools_for_prompt


_PROMPT = load_prompt("planner")


# ---------------------------------------------------------------------------
# P22: Retrieval Scout — pre-retrieval to inform planner
# ---------------------------------------------------------------------------

def retrieval_scout_node(state: AgentState) -> dict:
    """Lightweight pre-retrieval: run query against all docs, return top-3 candidate docs.

    The planner consumes scout_candidates to set target_doc intelligently,
    fixing the "blind planner" problem where it could only guess from doc_id strings.
    """
    query = state.get("query", "")
    if not query:
        return {"scout_candidates": []}

    try:
        from tools.colpali_tool import colpali_retrieve, _ensure_loaded, _encode_query, _maxsim, _state as _tool_state

        if not _ensure_loaded():
            logger.info("retrieval_scout: no indexes loaded, skipping")
            return {"scout_candidates": []}

        q_emb = _encode_query(query)

        # Per-doc best page: compute MaxSim for each doc, take max
        doc_scores: dict[str, dict] = {}
        indexes = _tool_state.get("indexes") or {}
        for doc_id, idx in indexes.items():
            scores = _maxsim(q_emb, idx["embeddings"])
            best_idx = scores.argmax().item()
            best_score = scores[best_idx].item()
            best_page = int(idx["page_nums"][best_idx])
            doc_scores[doc_id] = {
                "doc_id": doc_id,
                "top_page_num": best_page,
                "top_score": round(best_score, 4),
            }

        # Top-3 docs by best page score
        top_docs = sorted(doc_scores.values(), key=lambda d: d["top_score"], reverse=True)[:3]
        logger.info(f"retrieval_scout: top-3 candidate docs: {[(d['doc_id'], d['top_score']) for d in top_docs]}")
        return {"scout_candidates": top_docs}

    except Exception as e:
        logger.warning(f"retrieval_scout failed ({e}); planner will use doc_metadata only")
        return {"scout_candidates": []}


# ---------------------------------------------------------------------------
# Planner body
# ---------------------------------------------------------------------------

def _load_doc_metadata() -> str:
    """Load basic doc metadata from doc_memory.json (fallback when scout fails)."""
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


def _render_candidate_docs(state: AgentState) -> str:
    """P22: Render scout_candidates as richer doc metadata for the planner prompt."""
    candidates = state.get("scout_candidates") or []
    if not candidates:
        # Fall back to basic doc_metadata
        return _load_doc_metadata()

    lines: list[str] = []
    for c in candidates:
        doc_id = c.get("doc_id", "?")
        page = c.get("top_page_num", "?")
        score = c.get("top_score", 0)
        lines.append(f"- doc_id={doc_id}, top_hit_page={page}, relevance={score:.4f}")

    # Also append any docs NOT in scout (low relevance docs)
    all_ids = {c["doc_id"] for c in candidates}
    mem_path = INDEX_DIR / "doc_memory.json"
    if mem_path.exists():
        try:
            data = json.loads(mem_path.read_text(encoding="utf-8"))
            for d in data.get("docs") or []:
                if d["doc_id"] not in all_ids:
                    lines.append(f"- doc_id={d['doc_id']}, pages={d['page_count']} (low relevance)")
        except Exception:
            pass

    return "\n".join(lines)


def _render_history(state: AgentState, max_chars_per_turn: int = 200) -> str:
    """Render chat_history as compact bullet lines for the planner prompt.

    Each entry is {"role": "user"|"assistant", "content": str}. Long
    assistant answers are truncated — coreference resolution only needs the
    entities mentioned, not the full citation list.
    """
    history = state.get("chat_history") or []
    if not history:
        return "(no prior turns)"
    lines: list[str] = []
    for turn in history:
        role = turn.get("role", "?")
        content = (turn.get("content") or "").strip().replace("\n", " ")
        if len(content) > max_chars_per_turn:
            content = content[:max_chars_per_turn] + "…"
        prefix = "U" if role == "user" else "A"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def planner_node(state: AgentState) -> dict:
    """Call LLM to decompose the user query into an ordered list of SubTasks.

    Falls back to a single-SubTask plan (sub_query = original query) when
    the LLM key is missing or the call fails.
    """
    if not has_llm_key():
        logger.warning("DEEPSEEK_API_KEY not set — planner falls back to single sub-task")
        return _fallback(state)

    try:
        llm = get_llm("planner").with_structured_output(PlannerOutput, method="json_mode")
        # P22: use richer candidate_docs when scout ran; fall back to basic metadata
        doc_metadata = _render_candidate_docs(state)
        prompt = _PROMPT.format(
            query=state["query"],
            doc_metadata=doc_metadata,
            chat_history=_render_history(state),
            available_tools=get_tools_for_prompt(),
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
    """Return a single-SubTask plan using the raw user query."""
    return {"plan": [SubTask(sub_query=state["query"])], "plan_cursor": 0}
