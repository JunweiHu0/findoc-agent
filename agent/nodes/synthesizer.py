from __future__ import annotations

from loguru import logger

from ..llm import get_llm, has_llm_key
from ..prompts import load_prompt
from ..state import AgentState, Citation


_PROMPT = load_prompt("synthesizer")


def synthesizer_node(state: AgentState) -> dict:
    facts = state.get("extracted_facts") or []
    cvs = state.get("computed_values") or []
    citations = _dedupe_citations(facts)

    if not has_llm_key():
        logger.warning("DEEPSEEK_API_KEY not set — synthesizer emits stub answer")
        return {"answer": _stub_answer(state, facts, cvs), "citations": citations}

    try:
        llm = get_llm("synthesizer")
        prompt = _PROMPT.format(query=state["query"], evidence=_render_evidence(facts, cvs))
        msg = llm.invoke(prompt)
        return {"answer": msg.content, "citations": citations}
    except Exception as e:
        logger.warning(f"synthesizer LLM call failed ({e}); using stub answer")
        return {"answer": _stub_answer(state, facts, cvs), "citations": citations}


def _dedupe_citations(facts) -> list[Citation]:
    seen: set[tuple[str, int]] = set()
    out: list[Citation] = []
    for f in facts:
        key = (f.source_doc, f.source_page)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(doc_id=f.source_doc, page_num=f.source_page))
    return out


def _render_evidence(facts, cvs) -> str:
    if not facts and not cvs:
        return "(no evidence)"
    lines = []
    for f in facts:
        lines.append(f"- doc={f.source_doc} page={f.source_page}: {f.text}")
    for c in cvs:
        lines.append(f"- compute: {c.expr} = {c.value}")
    return "\n".join(lines)


def _stub_answer(state: AgentState, facts, cvs) -> str:
    parts = [f"Q: {state['query']}", ""]
    if not facts and not cvs:
        parts.append("[stub] no evidence collected")
        return "\n".join(parts)
    for f in facts:
        parts.append(f"- {f.text} [{f.source_doc} p.{f.source_page}]")
    for c in cvs:
        parts.append(f"- {c.expr} = {c.value} [calc]")
    return "\n".join(parts)
