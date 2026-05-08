"""Synthesizer node — composes final answer with streaming token output / 合成节点——汇总证据生成带引用答案，支持流式输出。

Citations are extracted by parsing `[doc_id p.N]` patterns from the generated answer text,
not by dumping every retrieved page — so the returned citations reflect what the model
actually used. 引用通过解析答案中的 [doc_id p.N] 模式得到，反映模型真实使用的页面。"""

from __future__ import annotations

import re

from loguru import logger

from ..llm import get_llm, has_llm_key
from ..prompts import load_prompt
from ..state import AgentState, Citation


_PROMPT = load_prompt("synthesizer")

# Matches inline citations like [doc_id p.23] or [doc_id p.1] in the answer.
# 匹配答案中的 [doc_id p.N] 内联引用
_CITATION_RE = re.compile(r"\[(\w+)\s+p\.(\d+)\]")

# Token hook — set by the backend to push streaming tokens out as SSE.
# Signature: callback(token: str) -> None. Called once per chunk during
# llm.stream(). When None, synthesizer falls back to a single invoke().
_token_hook: "callable | None" = None


def set_token_hook(hook: "callable | None") -> None:
    """Register a streaming token callback / 注册流式 token 回调，由后端 server 调用。"""
    global _token_hook
    _token_hook = hook


def _push_token(token: str) -> None:
    """Push a single token chunk to the registered hook / 推送单个 token 到注册的 hook。"""
    if _token_hook and token:
        try:
            _token_hook(token)
        except Exception:
            pass


def synthesizer_node(state: AgentState) -> dict:
    """Generate the final answer; citations are extracted from the answer text itself.
    生成最终答案，引用从答案文本中解析得到（仅模型真实使用的页面）。
    Streams token-by-token when hook is registered. Direct-answer mode (no retrieval)
    is signalled by `needs_retrieval=False` and uses a chat-only prompt variant.
    流式输出 token；直接回答模式由 needs_retrieval=False 触发，使用 chat-only 变体。"""
    facts = state.get("extracted_facts") or []
    cvs = state.get("computed_values") or []
    needs_retrieval = state.get("needs_retrieval", True)

    if not has_llm_key():
        logger.warning("DEEPSEEK_API_KEY not set — synthesizer emits stub answer")
        answer = _stub_answer(state, facts, cvs)
        _push_token(answer)  # let UI still see something arrive incrementally
        return {"answer": answer, "citations": _citations_from_answer(answer, facts)}

    # Pick variant: direct (no retrieval) > numeric > base.
    # 选模板：direct（无检索）> numeric > base
    if not needs_retrieval:
        syn_variant = "direct"
    else:
        qc = state.get("query_class", "")
        syn_variant = "numeric" if qc in ("multi_step_calc", "cross_doc_compare") else "base"
    try:
        variant_prompt = load_prompt("synthesizer", variant=syn_variant)
    except Exception:
        variant_prompt = _PROMPT

    try:
        llm = get_llm("synthesizer")
        if not needs_retrieval:
            chat_history = _render_chat_history(state.get("chat_history") or [])
            prompt = variant_prompt.format(query=state["query"], chat_history=chat_history)
        else:
            prompt = variant_prompt.format(query=state["query"], evidence=_render_evidence(facts, cvs))

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
            return {"answer": content, "citations": _citations_from_answer(content, facts)}

        msg = llm.invoke(prompt)
        return {"answer": msg.content, "citations": _citations_from_answer(msg.content, facts)}
    except Exception as e:
        logger.warning(f"synthesizer LLM call failed ({e}); using stub answer")
        answer = _stub_answer(state, facts, cvs)
        _push_token(answer)
        return {"answer": answer, "citations": _citations_from_answer(answer, facts)}


def _citations_from_answer(answer: str, facts) -> list[Citation]:
    """Extract citations the model actually wrote into the answer.
    Validate against retrieved facts: only keep (doc_id, page) pairs that exist in evidence.
    从答案中抽取模型真正写出的引用，并用证据集合做有效性过滤——剥离虚构引用。"""
    if not answer:
        return []
    valid: set[tuple[str, int]] = {(f.source_doc, f.source_page) for f in (facts or [])}
    seen: set[tuple[str, int]] = set()
    out: list[Citation] = []
    for m in _CITATION_RE.finditer(answer):
        doc_id = m.group(1)
        try:
            page_num = int(m.group(2))
        except ValueError:
            continue
        key = (doc_id, page_num)
        if key in seen:
            continue
        # If we have evidence at all, require the citation to exist in it.
        # 有证据时，只保留证据集中存在的引用
        if valid and key not in valid:
            continue
        seen.add(key)
        out.append(Citation(doc_id=doc_id, page_num=page_num))
    return out


def _render_chat_history(history: list[dict]) -> str:
    """Format chat history for direct-answer mode / 直接回答模式渲染对话历史。"""
    if not history:
        return "(no prior turns)"
    lines = []
    for turn in history[-6:]:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        prefix = "用户" if role == "user" else "助手"
        lines.append(f"{prefix}: {content[:300]}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _render_evidence(facts, cvs) -> str:
    """Format facts and computed values as bullet lines for the LLM prompt / 格式化事实和计算值为 LLM prompt 的列表行。"""
    if not facts and not cvs:
        return "(no evidence)"
    lines = []
    for f in facts:
        lines.append(f"- doc={f.source_doc} page={f.source_page}: {f.text}")
    for c in cvs:
        lines.append(f"- compute: {c.expr} = {c.value}")
    return "\n".join(lines)


def _stub_answer(state: AgentState, facts, cvs) -> str:
    """Build a stub answer from raw evidence when LLM is unavailable / LLM 不可用时用原始证据构建 stub 回答。"""
    parts = [f"Q: {state['query']}", ""]
    if not facts and not cvs:
        parts.append("[stub] no evidence collected")
        return "\n".join(parts)
    for f in facts:
        parts.append(f"- {f.text} [{f.source_doc} p.{f.source_page}]")
    for c in cvs:
        parts.append(f"- {c.expr} = {c.value} [calc]")
    return "\n".join(parts)
