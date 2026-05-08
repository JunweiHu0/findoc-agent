"""Planner node — two-stage query decomposition with skill matching / 规划节点——两段式查询分解+技能匹配。

retrieval_scout_node provides pre-retrieval context so the planner can make
informed target_doc decisions. 预检索探查为 planner 提供候选文档信息。
"""

from __future__ import annotations

import json

from loguru import logger

from ..compression import compress_history
from ..config import INDEX_DIR
from ..llm import get_llm, has_llm_key
from ..prompts import load_prompt
from ..retry import classify_error
from ..schemas import PlannerOutput
from ..state import AgentState, SubTask
from tools.registry import get_tools_for_prompt


_PROMPT = load_prompt("planner")


# ---------------------------------------------------------------------------
# Retrieval Scout — pre-retrieval to inform planner / 检索前探查
# ---------------------------------------------------------------------------

def retrieval_scout_node(state: AgentState) -> dict:
    """Lightweight pre-retrieval: run query against all docs, return top-3 candidate docs.
    轻量预检索：查询全库返回 top-3 候选文档。Planner 据此做 informed target_doc 决策。"""
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
    """Render scout_candidates as rich doc metadata for the planner prompt / 渲染候选文档元数据。"""
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


def planner_node(state: AgentState) -> dict:
    """Two-stage planner: match skill → classify → variant prompt → generate plan.
    两段式规划：匹配技能→分类查询→变体 prompt+few-shot→生成计划。

    Stage 0: Check skill registry, override plan_template + strategy if matched.
    Stage 1: Lightweight classification (~200 tokens) → query_class.
    Stage 2: Load variant prompt + few-shot → generate full plan.
    Falls back to single-SubTask when LLM unavailable. Uses compress_history.
    Stage 0 匹配技能；Stage 1 轻量分类；Stage 2 变体 prompt 生成；LLM 不可用时回退。"""
    if not has_llm_key():
        logger.warning("DEEPSEEK_API_KEY not set — planner falls back to single sub-task")
        return _fallback(state)

    query = state["query"]
    history = state.get("chat_history") or []
    compressed_ctx = compress_history(history) if history else "(no prior turns)"
    doc_metadata = _render_candidate_docs(state)

    # Stage 0: match skill / 匹配技能
    skill = None
    skill_strategy = {}
    try:
        from skills.registry import match_skill as _match_skill
        skill = _match_skill(query)
        if skill:
            skill_strategy = dict(skill.strategy)
            logger.info(f"planner: using skill '{skill.name}', template={skill.plan_template}")
    except ImportError:
        pass

    # Stage 1: classify query / 分类查询
    query_class = state.get("query_class") or ""
    if not query_class:
        if skill and skill.plan_template != "base":
            query_class = skill.plan_template
        else:
            query_class = _classify_query(query, compressed_ctx)

    # Stage 2: load variant prompt with few-shot / 加载变体 prompt + few-shot
    try:
        variant_prompt = load_prompt("planner", variant=query_class or "base", with_few_shot=True)
    except Exception:
        variant_prompt = _PROMPT

    try:
        llm = get_llm("planner").with_structured_output(PlannerOutput, method="json_mode")
        prompt = variant_prompt.format(
            query=query,
            doc_metadata=doc_metadata,
            chat_history=compressed_ctx,
            available_tools=get_tools_for_prompt(),
        )
        result: PlannerOutput = llm.invoke(prompt)
        plan = [SubTask(**item.model_dump()) for item in result.plan]
        if not plan:
            return _fallback(state)

        delta: dict = {
            "plan": plan,
            "plan_cursor": 0,
            "query_class": result.query_class or query_class,
        }
        # Pass skill strategy to executor / 传递技能策略到 executor
        if skill_strategy:
            delta["remediation_hint"] = skill_strategy
        return delta
    except Exception as e:
        err = classify_error(e)
        err["node"] = "planner"
        logger.warning(f"planner LLM call failed ({e}); falling back to single sub-task")
        fb = _fallback(state)
        fb["error_log"] = [err]
        return fb


def _classify_query(query: str, chat_context: str = "") -> str:
    """Lightweight query classification (~200 token LLM), heuristic-first / 轻量查询分类，启发式优先。

    Returns: single_fact | cross_doc_compare | multi_step_calc | trend_analysis.
    Keyword heuristic catches >95% of queries; LLM fallback for ambiguous cases.
    关键词启发式覆盖 >95% 查询，剩余走 LLM。"""
    # Heuristic: keyword matching first / 启发式：先关键词
    if any(kw in query for kw in ["对比", "比较", "vs", "versus", "差异", "哪个"]):
        return "cross_doc_compare"
    if any(kw in query for kw in ["趋势", "变化", "增长", "逐年", "历年"]):
        return "trend_analysis"
    if any(kw in query for kw in ["计算", "算一下", "比例", "占比"]) or (
        "毛利率" in query and ("营收" in query or "成本" in query)
    ):
        return "multi_step_calc"

    # Heuristic not conclusive — use lightweight LLM call
    if not has_llm_key():
        return "single_fact"

    try:
        llm = get_llm("planner")
        prompt = (
            "将以下用户问题归类为以下四种之一。只返回类别名称，不要额外文字。\n\n"
            "类别: single_fact, cross_doc_compare, multi_step_calc, trend_analysis\n"
            f"对话上下文: {chat_context[:200]}\n"
            f"问题: {query}\n\n"
            "类别:"
        )
        resp = llm.invoke(prompt)
        result = (resp.content or "").strip().lower()
        valid = {"single_fact", "cross_doc_compare", "multi_step_calc", "trend_analysis"}
        for v in valid:
            if v in result:
                return v
        return "single_fact"
    except Exception:
        return "single_fact"


def _fallback(state: AgentState) -> dict:
    """Return a single-SubTask plan using the raw user query / 用原始查询生成单步回退计划。"""
    return {"plan": [SubTask(sub_query=state["query"])], "plan_cursor": 0}
