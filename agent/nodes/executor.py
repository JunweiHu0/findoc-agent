"""Executor node — dispatches sub-tasks via tool registry (P21) or legacy schema routing.

For each SubTask in the plan (at plan_cursor), the executor either:
- dispatches explicit tool_calls through the registry (P21 path), or
- routes by expected_output_schema: number -> calculator, other -> retrieval+VLM.

VLM reads are run concurrently via ThreadPoolExecutor (max 5 workers).
P25 cross-turn fact cache is checked before any retrieval.
P24 structured fact extraction runs after every VLM read.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from tools.calculator import calculate
from tools.colpali_tool import colpali_retrieve
from tools.registry import dispatch as _registry_dispatch, REGISTRY
from tools.vlm_tool import vlm_read_page

from ..config import TOP_K
from ..state import AgentState, ComputedValue, Fact


def executor_node(state: AgentState) -> dict:
    """Process the current SubTask at plan_cursor and advance the cursor by 1."""
    cursor = state.get("plan_cursor", 0)
    plan = state.get("plan") or []
    if cursor >= len(plan):
        return {}

    sub_task = plan[cursor]

    # P21: tool_calls takes priority over legacy expected_output_schema
    if sub_task.tool_calls:
        return _run_tool_calls(sub_task, cursor, state)

    if sub_task.expected_output_schema == "number":
        return _run_calculation(sub_task, cursor)
    return _run_retrieval(sub_task, cursor, state.get("doc_filter"), state)


def _run_tool_calls(sub_task, cursor: int, state: AgentState) -> dict:
    """P21: Dispatch each ToolCall via the registry. Accumulate facts/pages/computed values."""
    facts: list[Fact] = []
    pages: list = []
    cvs: list[ComputedValue] = []

    for tc in sub_task.tool_calls:
        tool_name = tc.get("tool", "") if isinstance(tc, dict) else getattr(tc, "tool", "")
        args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})

        if not tool_name:
            logger.warning("tool_calls entry missing 'tool' key; skipping")
            continue

        try:
            result = _registry_dispatch(tool_name, args)
        except KeyError as e:
            logger.warning(f"Unknown tool '{tool_name}' — falling back to legacy routing: {e}")
            return _run_retrieval(sub_task, cursor, state.get("doc_filter"), state)
        except Exception as e:
            logger.warning(f"Tool '{tool_name}' failed ({e}); skipping this tool call")
            continue

        # Collect results based on tool category
        spec = REGISTRY.get(tool_name)
        if spec and spec.category == "retrieval":
            result_pages = _extract_pages(result)
            pages.extend(result_pages)
            # Auto-read pages with VLM if tool was retrieve_pages
            if result_pages:
                with ThreadPoolExecutor(max_workers=min(len(result_pages), 5)) as ex:
                    texts = list(ex.map(
                        lambda h: vlm_read_page(
                            h.get("image_path", ""),
                            instruction=sub_task.sub_query,
                        ),
                        result_pages,
                    ))
                for hit, text in zip(result_pages, texts):
                    facts.append(Fact(
                        text=text,
                        source_doc=hit.get("doc_id", ""),
                        source_page=hit.get("page_num", 0),
                        sub_task_idx=cursor,
                    ))
        elif spec and spec.category == "compute":
            val = _extract_value(result)
            cvs.append(ComputedValue(expr=str(args), value=val, sub_task_idx=cursor))
        elif spec and spec.category == "reading":
            text = _extract_text(result)
            facts.append(Fact(
                text=text,
                source_doc=args.get("doc_id", "unknown"),
                source_page=args.get("page_num", 0),
                sub_task_idx=cursor,
            ))
        elif spec and spec.category == "resolution":
            # disambiguate_caliber: store results as facts
            auth = result.get("authoritative_fact", {}) if isinstance(result, dict) else {}
            explanation = result.get("explanation", "") if isinstance(result, dict) else ""
            facts.append(Fact(
                text=f"[口径消歧] {explanation}",
                source_doc=auth.get("doc_id", "unknown"),
                source_page=auth.get("page_num", 0),
                sub_task_idx=cursor,
            ))

    return {
        "retrieved_pages": pages,
        "extracted_facts": facts,
        "computed_values": cvs,
        "plan_cursor": cursor + 1,
    }


def _extract_pages(result) -> list[dict]:
    """Extract page hits from a tool result (dict, list, or pydantic model)."""
    if isinstance(result, dict):
        return result.get("pages") or []
    if isinstance(result, list):
        return result
    if hasattr(result, "pages"):
        return [p.model_dump() if hasattr(p, "model_dump") else p for p in result.pages]
    return []


def _extract_value(result) -> float:
    """Extract a float value from a calculator tool result."""
    if isinstance(result, dict):
        return float(result.get("value", float("nan")))
    if hasattr(result, "value"):
        return float(result.value)
    return float(result) if isinstance(result, (int, float)) else float("nan")


def _extract_text(result) -> str:
    """Extract text content from a VLM reading tool result."""
    if isinstance(result, dict):
        return result.get("extracted_text") or result.get("text") or str(result)
    if hasattr(result, "extracted_text"):
        return result.extracted_text
    return str(result)


def _run_calculation(sub_task, cursor: int) -> dict:
    """Execute a 'number' schema sub-task via the AST-safe calculator."""
    try:
        value = calculate(sub_task.sub_query)
    except Exception as e:
        logger.warning(f"calculator rejected '{sub_task.sub_query}': {e}")
        value = float("nan")
    cv = ComputedValue(expr=sub_task.sub_query, value=value, sub_task_idx=cursor)
    return {"computed_values": [cv], "plan_cursor": cursor + 1}


def _run_retrieval(sub_task, cursor: int, session_doc_filter: list[str] | None, state: AgentState | None = None) -> dict:
    """Run ColQwen retrieval + concurrent VLM page reads for a text/table sub-task.

    Checks cross-turn fact cache (P25) before retrieval. Applies P20 remediation
    hints (widened top_k). Runs VLM on retrieved pages concurrently (max 5 workers).
    """
    if sub_task.target_doc:
        doc_filter = [sub_task.target_doc]
    elif session_doc_filter:
        doc_filter = session_doc_filter
    else:
        doc_filter = None

    # P20: if remediation widened top_k, use it
    hint = (state or {}).get("remediation_hint") or {}
    top_k = hint.get("widened_top_k", TOP_K)

    # P25: check known_facts before retrieval
    if state:
        known = state.get("known_facts") or []
        if known:
            cached_facts, cached_pages = _check_known_facts(sub_task.sub_query, known, cursor)
            if cached_facts:
                logger.info(f"executor: {len(cached_facts)} fact(s) served from cross-turn cache")
                return {
                    "retrieved_pages": cached_pages,
                    "extracted_facts": cached_facts,
                    "plan_cursor": cursor + 1,
                }

    hits = colpali_retrieve(sub_task.sub_query, top_k=top_k, doc_filter=doc_filter)

    if not hits:
        texts: list[str] = []
    else:
        with ThreadPoolExecutor(max_workers=min(len(hits), 5)) as ex:
            texts = list(ex.map(
                lambda h: vlm_read_page(h.image_path or "", instruction=sub_task.sub_query),
                hits,
            ))

    facts: list[Fact] = []
    for hit, text in zip(hits, texts):
        facts.append(Fact(
            text=text,
            source_doc=hit.doc_id,
            source_page=hit.page_num,
            sub_task_idx=cursor,
        ))

    # P24: run structured fact extraction after VLM
    facts = _maybe_extract_structured(facts)

    return {
        "retrieved_pages": hits,
        "extracted_facts": facts,
        "plan_cursor": cursor + 1,
    }


# ---------------------------------------------------------------------------
# P24: Structured fact extraction
# ---------------------------------------------------------------------------

def _maybe_extract_structured(facts: list[Fact]) -> list[Fact]:
    """Apply regex-based structured extraction to each Fact (P24). Falls back to original facts on error."""
    try:
        from tools.fact_extractor import extract_structured_facts
        return extract_structured_facts(facts)
    except ImportError:
        return facts
    except Exception as e:
        logger.debug(f"structured fact extraction skipped: {e}")
        return facts


# ---------------------------------------------------------------------------
# P25: Cross-turn fact cache lookup
# ---------------------------------------------------------------------------

def _check_known_facts(sub_query: str, known_facts: list[dict], cursor: int) -> tuple[list[Fact], list]:
    """Check if known_facts (from prior turns) can satisfy this sub_query.

    Uses simple keyword overlap heuristic: sub_query must contain entity + period
    + metric hints from the known fact. Returns (facts, pages) if matched, else ([], []).
    """
    from agent.state import PageHit

    matched_facts: list[Fact] = []
    matched_pages: list[PageHit] = []

    for kf in known_facts:
        entity = kf.get("entity") or ""
        period = kf.get("period") or ""
        metric = kf.get("metric") or ""

        # Check keyword overlap: all non-empty keys should appear in sub_query
        hits = 0
        total = 0
        for keyword in [entity, period, metric]:
            if keyword:
                total += 1
                if keyword in sub_query:
                    hits += 1

        if total > 0 and hits >= total * 0.5:  # at least 50% keyword match
            matched_facts.append(Fact(
                text=kf.get("text", ""),
                source_doc=kf.get("source_doc", "cached(conv)"),
                source_page=kf.get("source_page", 0),
                sub_task_idx=cursor,
                entity=entity,
                period=period,
                metric=metric,
                value=kf.get("value"),
                unit=kf.get("unit"),
                raw_kind="numeric" if kf.get("value") is not None else "unstructured",
            ))
            matched_pages.append(PageHit(
                doc_id=kf.get("source_doc", "cached(conv)"),
                page_num=kf.get("source_page", 0),
                score=1.0,
            ))

    return matched_facts, matched_pages
