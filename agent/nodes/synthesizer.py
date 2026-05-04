from __future__ import annotations

from loguru import logger

from ..llm import get_llm, has_llm_key
from ..prompts import load_prompt
from ..state import AgentState, Citation


_PROMPT = load_prompt("synthesizer")

# P16: token hook — set by the backend to push streaming tokens out as SSE.
# Signature: callback(token: str) -> None. Called once per chunk during
# llm.stream(). When None, synthesizer falls back to a single invoke().
_token_hook: "callable | None" = None


def set_token_hook(hook: "callable | None") -> None:
    global _token_hook
    _token_hook = hook


def _push_token(token: str) -> None:
    if _token_hook and token:
        try:
            _token_hook(token)
        except Exception:
            pass


def synthesizer_node(state: AgentState) -> dict:
    facts = state.get("extracted_facts") or []
    cvs = state.get("computed_values") or []
    citations = _dedupe_citations(facts)

    if not has_llm_key():
        logger.warning("DEEPSEEK_API_KEY not set — synthesizer emits stub answer")
        answer = _stub_answer(state, facts, cvs)
        _push_token(answer)  # let UI still see something arrive incrementally
        return {"answer": answer, "citations": citations}

    try:
        llm = get_llm("synthesizer")
        prompt = _PROMPT.format(query=state["query"], evidence=_render_evidence(facts, cvs))

        # If a token hook is registered, stream chunks; otherwise fall back to invoke().
        if _token_hook is not None:
            buf: list[str] = []
            for chunk in llm.stream(prompt):
                piece = getattr(chunk, "content", "") or ""
                if piece:
                    buf.append(piece)
                    _push_token(piece)
            content = "".join(buf)
            if not content:
                # Empty stream — fall back to a non-streaming retry to avoid blank answer.
                content = llm.invoke(prompt).content
            return {"answer": content, "citations": citations}

        msg = llm.invoke(prompt)
        return {"answer": msg.content, "citations": citations}
    except Exception as e:
        logger.warning(f"synthesizer LLM call failed ({e}); using stub answer")
        answer = _stub_answer(state, facts, cvs)
        _push_token(answer)
        return {"answer": answer, "citations": citations}


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
