"""Gradio demo for FinDoc Agent — DeepSeek-style 3-column workspace.

Layout (left → right):
- Left rail: "新建对话" + clickable list of past conversations (per-session
  state via ``gr.State``; the list is rendered with ``@gr.render`` so it
  reacts to state changes without manual diffing).
- Center: hero/chip empty state → bubble chat → rounded input.
- Right rail: tabs for 执行轨迹 / 工具调用 / 引用页 / 文档库.

Each conversation owns its own ``history`` (chatbot messages), ``trace``
(last AgentState snapshot for the trace tab), ``citations`` (gallery),
and ``tool_log`` (per-node summary entries). Switching conversations
swaps all four panes from that conversation's stored snapshot.

Streaming: ``compile_graph().stream(stream_mode="updates")`` is wrapped
by ``_stream_agent`` which yields ``(status, accum_state, node, delta)``
per node so the chat handler can refresh the chatbot, trace, and tool
log in lockstep instead of blocking on a single ``.invoke()``.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import gradio as gr

from agent.config import INDEX_DIR, PAGES_DIR
from agent.graph import compile_graph
from agent.state import Citation


_app = compile_graph()

_NODE_STATUS = {
    "planner": "📝 拆解任务…",
    "executor": "🔧 检索 & 解读…",
    "verifier": "🔍 反思校验…",
    "synthesizer": "✍️ 生成答案…",
}

_NODE_LABELS = {
    "planner": "Planner · 任务规划",
    "executor": "Executor · 检索与解读",
    "verifier": "Verifier · 反思校验",
    "synthesizer": "Synthesizer · 答案合成",
}

_LIST_REDUCED_KEYS = ("retrieved_pages", "extracted_facts", "computed_values")

_SUGGESTIONS = [
    "📊 贵州茅台 2023 年的营业收入是多少？",
    "🆚 对比贵州茅台、宁德时代 2023 年的毛利率",
    "👥 招商银行 2024 年末员工总数是多少？",
    "🧪 恒瑞医药 2024 年研发投入占营业收入的比例",
]


# ---------------------------------------------------------------------------
# CSS — white base with warm-amber accents (DeepSeek-ish three-column shell).
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ---------- Reset ---------- */
footer { display: none !important; }
.gradio-container {
    padding: 0 !important;
    max-width: 100% !important;
    background: #ffffff !important;
}
.app.svelte-1eyzfp4, .app { padding: 0 !important; }
* { box-sizing: border-box; }

/* ---------- Palette ---------- */
:root {
    --accent: #d97706;          /* amber-600 */
    --accent-hover: #b45309;    /* amber-700 */
    --accent-soft: #fef3c7;     /* amber-100 */
    --accent-tint: #fffbeb;     /* amber-50  */
    --warm-border: #fde68a;     /* amber-200 */
    --warm-bg: #fefce8;         /* yellow-50 */
    --ink: #292524;             /* stone-800 */
    --ink-soft: #57534e;        /* stone-600 */
    --ink-muted: #a8a29e;       /* stone-400 */
    --line: #e7e5e4;            /* stone-200 */
}

/* ---------- Header ---------- */
#app-header {
    padding: 12px 20px !important;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(90deg, #ffffff 0%, var(--accent-tint) 70%, var(--accent-soft) 100%);
    margin: 0 !important;
    align-items: center !important;
}
#app-header > * { padding: 0 !important; }
#app-header h1 {
    margin: 0 !important;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--accent);
    display: inline-block;
}
#app-header h1::before {
    content: "◆ ";
    color: var(--accent);
    margin-right: 2px;
}
#app-header .subtitle {
    font-size: 12px;
    color: var(--ink-soft);
    margin-left: 12px;
}

/* ---------- Three-column body ---------- */
#body-row { gap: 0 !important; min-height: calc(100vh - 50px); }

/* ---------- Left sidebar (conversation history) ---------- */
#left-sidebar {
    background: #fafaf9 !important;
    border-right: 1px solid var(--line) !important;
    padding: 14px 10px !important;
    gap: 6px !important;
    min-height: calc(100vh - 50px);
    overflow-y: auto;
}
#new-chat-btn {
    background: var(--accent) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 4px !important;
    box-shadow: 0 2px 6px rgba(217,119,6,0.20) !important;
    transition: all 0.15s !important;
}
#new-chat-btn:hover {
    background: var(--accent-hover) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(217,119,6,0.28) !important;
}

.section-title {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--ink-muted);
    margin: 14px 6px 4px;
}

#conv-list { gap: 1px !important; padding: 2px 0; }
#conv-list .empty-hint { padding: 10px 8px; color: var(--ink-muted); font-size: 12px; font-style: italic; }
.conv-row {
    margin: 1px 0 !important;
    gap: 2px !important;
    align-items: center;
}
.conv-btn {
    background: transparent !important;
    border: 1px solid transparent !important;
    text-align: left !important;
    padding: 8px 10px !important;
    font-size: 13px !important;
    color: var(--ink-soft) !important;
    border-radius: 8px !important;
    line-height: 1.35 !important;
    flex: 1 1 auto !important;
    white-space: normal !important;
    box-shadow: none !important;
    height: auto !important;
    min-height: 32px !important;
    font-weight: 400 !important;
}
.conv-btn:hover {
    background: var(--accent-tint) !important;
    color: var(--ink) !important;
}
.conv-active .conv-btn {
    background: var(--accent-soft) !important;
    border-color: var(--warm-border) !important;
    font-weight: 600 !important;
    color: var(--accent-hover) !important;
}
.conv-del {
    background: transparent !important;
    border: none !important;
    color: var(--ink-muted) !important;
    font-size: 16px !important;
    padding: 2px 6px !important;
    min-width: 22px !important;
    width: 22px !important;
    height: 24px !important;
    box-shadow: none !important;
    line-height: 1 !important;
    flex: 0 0 22px !important;
    border-radius: 6px !important;
}
.conv-del:hover {
    color: #ef4444 !important;
    background: #fef2f2 !important;
}

/* ---------- Center column ---------- */
#chat-col {
    background: #ffffff;
    padding: 0 !important;
    overflow: hidden;
}
#chat-inner {
    max-width: 820px;
    margin: 0 auto;
    padding: 18px 16px 12px;
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

/* Hero (empty state) */
#hero-html {
    text-align: center;
    padding: 56px 16px 8px;
}
#hero-html .badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 999px;
    background: var(--accent-soft);
    border: 1px solid var(--warm-border);
    font-size: 11px;
    color: var(--accent-hover);
    margin-bottom: 18px;
    letter-spacing: 0.05em;
    font-weight: 600;
}
#hero-html h2 {
    font-size: 32px !important;
    font-weight: 700 !important;
    margin: 0 0 12px !important;
    color: var(--ink);
    line-height: 1.2;
    letter-spacing: -0.01em;
}
#hero-html h2 .y { color: var(--accent); }
#hero-html p {
    font-size: 14px !important;
    color: var(--ink-soft);
    margin: 0 auto !important;
    max-width: 540px;
    line-height: 1.65;
}

#chips-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px !important;
    margin: 18px 0 8px;
}
#chips-grid > div { flex: none !important; min-width: 0 !important; }
#chips-grid button {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    text-align: left !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--ink) !important;
    height: auto !important;
    min-height: 60px !important;
    line-height: 1.5 !important;
    white-space: normal !important;
    transition: all 0.15s !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
}
#chips-grid button:hover {
    border-color: var(--accent) !important;
    background: var(--accent-tint) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(217,119,6,0.10) !important;
}

/* Chatbot — bubble layout, white/amber palette */
#chatbot {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    flex-grow: 1;
}
#chatbot .wrap, #chatbot .message-wrap, #chatbot .bubble-wrap {
    background: transparent !important;
}
#chatbot .message {
    border-radius: 14px !important;
    padding: 12px 16px !important;
    line-height: 1.7 !important;
    font-size: 14.5px !important;
}
#chatbot .user-row .message,
#chatbot div[data-testid="user"] .message {
    background: linear-gradient(135deg, #fef3c7, #fde68a) !important;
    border: 1px solid var(--warm-border) !important;
    color: var(--ink) !important;
}
#chatbot .bot-row .message,
#chatbot div[data-testid="bot"] .message {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
}
#chatbot code {
    font-size: 12.5px !important;
    background: var(--accent-soft) !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    color: var(--accent-hover) !important;
}

/* Input area */
#input-wrap { padding: 8px 0 4px; background: #ffffff; }
#msg-box {
    box-shadow: 0 4px 20px rgba(217,119,6,0.10);
    border-radius: 16px;
}
#msg-box > label { display: none !important; }
#msg-box textarea {
    border-radius: 16px !important;
    border: 1px solid var(--line) !important;
    background: #ffffff !important;
    padding: 14px 56px 14px 18px !important;
    font-size: 14.5px !important;
    line-height: 1.55 !important;
    min-height: 52px !important;
    max-height: 160px !important;
    resize: none !important;
    transition: all 0.2s !important;
    color: var(--ink) !important;
}
#msg-box textarea::placeholder { color: var(--ink-muted) !important; }
#msg-box textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(217,119,6,0.12) !important;
    outline: none !important;
}
#msg-box button {
    background: var(--accent) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
}
#msg-box button:hover { background: var(--accent-hover) !important; }

#status-line {
    text-align: center;
    font-size: 12px !important;
    color: var(--ink-soft) !important;
    padding: 6px 0 12px;
    min-height: 22px;
}
#status-line p { margin: 0 !important; font-size: 12px !important; }

/* ---------- Right sidebar (multi-function panel) ---------- */
#right-sidebar {
    background: var(--accent-tint) !important;
    border-left: 1px solid var(--warm-border) !important;
    padding: 0 !important;
    min-height: calc(100vh - 50px);
    overflow-y: auto;
}
#right-sidebar > .gradio-tabs { background: transparent; }
#right-sidebar .tab-nav {
    background: transparent !important;
    border-bottom: 1px solid var(--warm-border) !important;
    padding: 0 8px !important;
    margin: 0 !important;
    gap: 0 !important;
}
#right-sidebar .tab-nav button {
    background: transparent !important;
    color: var(--ink-soft) !important;
    border-radius: 0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 10px 10px !important;
    margin: 0 2px !important;
}
#right-sidebar .tab-nav button.selected {
    color: var(--accent-hover) !important;
    border-bottom-color: var(--accent) !important;
    font-weight: 700 !important;
    background: transparent !important;
}
#right-sidebar .tab-nav button:hover { color: var(--accent) !important; }
#right-sidebar .tabitem { padding: 14px !important; background: transparent !important; }

#right-trace-md, #right-tools-md, #right-doc-md {
    background: #ffffff;
    border: 1px solid var(--warm-border);
    border-radius: 10px;
    padding: 12px 14px !important;
}
#right-trace-md p, #right-trace-md li,
#right-tools-md p, #right-tools-md li,
#right-doc-md p, #right-doc-md li {
    font-size: 12.5px !important;
    line-height: 1.65 !important;
    color: var(--ink) !important;
}
#right-tools-md h4 {
    font-size: 12.5px !important;
    color: var(--accent-hover) !important;
    margin: 12px 0 4px !important;
    padding-bottom: 4px;
    border-bottom: 1px dashed var(--warm-border);
    font-weight: 700;
}
#right-tools-md h4:first-child { margin-top: 0 !important; }

#right-cite-strip {
    border: none !important;
    background: transparent !important;
}
#right-cite-strip .grid-wrap { gap: 6px !important; padding: 0 !important; }
#right-cite-strip .thumbnail-item {
    border-radius: 8px !important;
    border: 1px solid var(--warm-border) !important;
    overflow: hidden;
    transition: all 0.15s !important;
    background: #ffffff !important;
}
#right-cite-strip .thumbnail-item:hover {
    border-color: var(--accent) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 10px rgba(217,119,6,0.15);
}
#right-cite-empty {
    background: #ffffff;
    border: 1px dashed var(--warm-border);
    border-radius: 10px;
    padding: 24px;
    text-align: center;
    color: var(--ink-muted);
    font-size: 12.5px;
    font-style: italic;
}

#refresh-doc-btn {
    background: #ffffff !important;
    color: var(--accent-hover) !important;
    border: 1px solid var(--warm-border) !important;
    font-size: 12px !important;
    border-radius: 8px !important;
    margin-top: 8px !important;
}
#refresh-doc-btn:hover {
    background: var(--accent-soft) !important;
    border-color: var(--accent) !important;
}
"""


# ---------------------------------------------------------------------------
# Doc memory
# ---------------------------------------------------------------------------
def _load_doc_memory() -> list[dict]:
    mem_path = INDEX_DIR / "doc_memory.json"
    if not mem_path.exists():
        return []
    try:
        data = json.loads(mem_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data.get("docs") or []


def _doc_summary_md() -> str:
    docs = _load_doc_memory()
    if not docs:
        return "*尚未建立索引。*\n\n```\npython -m ingestion.build_index\n```"
    lines = ["**已索引文档**", ""]
    for d in docs:
        lines.append(f"- `{d['doc_id']}` · {d['page_count']} 页")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Citation rendering
# ---------------------------------------------------------------------------
def _resolve_page_image(doc_id: str, page_num: int) -> str | None:
    candidate = PAGES_DIR / doc_id / f"p{page_num:03d}.png"
    return str(candidate) if candidate.exists() else None


def _format_citations_inline(citations: list[Citation]) -> str:
    if not citations:
        return ""
    refs = " ".join(f"`[{c.doc_id} p.{c.page_num}]`" for c in citations)
    return f"\n\n**引用**：{refs}"


def _gallery_items(citations: list[Citation]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for c in citations:
        path = _resolve_page_image(c.doc_id, c.page_num)
        if path:
            items.append((path, f"{c.doc_id} p.{c.page_num}"))
    return items


# ---------------------------------------------------------------------------
# Trace + tool log rendering
# ---------------------------------------------------------------------------
def _render_trace(state: dict[str, Any]) -> str:
    if not state:
        return "_(尚无轨迹)_"

    sections: list[str] = []
    plan = state.get("plan") or []
    if plan:
        cursor = state.get("plan_cursor", 0)
        sections.append("**Plan**")
        for i, p in enumerate(plan):
            target = getattr(p, "target_doc", None) or "any"
            schema = getattr(p, "expected_output_schema", "text")
            sub_query = getattr(p, "sub_query", str(p))
            mark = "▶︎" if i == cursor else ("✓" if i < cursor else "·")
            sections.append(f"{mark} {i+1}. ({schema} · {target}) {sub_query}")
        sections.append(f"\n_cursor_: {cursor}/{len(plan)}")

    iter_count = state.get("reflexion_iter") or 0
    if iter_count > 0:
        sufficient = state.get("is_sufficient", False)
        marker = "✅ sufficient" if sufficient else "↻ continue"
        sections.append(f"\n**Reflexion** · iter={iter_count} · {marker}")
        missing = state.get("missing_info") or ""
        if missing and not sufficient:
            sections.append(f"&nbsp;&nbsp;_missing_: {missing}")

    pages = state.get("retrieved_pages") or []
    facts = state.get("extracted_facts") or []
    cvs = state.get("computed_values") or []
    if pages or facts or cvs:
        sections.append(
            f"\n**统计** · pages={len(pages)} · facts={len(facts)} · calc={len(cvs)}"
        )

    return "\n".join(sections) if sections else "_(尚无轨迹)_"


def _summarize_node(node: str, delta: dict[str, Any]) -> str:
    """Plain-language summary of what a node produced this step."""
    if node == "planner":
        plan = delta.get("plan") or []
        if not plan:
            return "_未生成计划_"
        return f"产出 **{len(plan)}** 步执行计划"
    if node == "executor":
        pages = len(delta.get("retrieved_pages") or [])
        facts = len(delta.get("extracted_facts") or [])
        cvs = len(delta.get("computed_values") or [])
        return f"检索页 **{pages}** · 抽取事实 **{facts}** · 计算 **{cvs}**"
    if node == "verifier":
        suff = delta.get("is_sufficient")
        miss = delta.get("missing_info") or ""
        if suff is True:
            return "✅ 信息充分，进入合成"
        if suff is False:
            return f"↻ 触发再检索 · _{miss}_" if miss else "↻ 触发再检索"
        return "_(verifier 完成)_"
    if node == "synthesizer":
        ans = delta.get("answer") or ""
        cites = len(delta.get("citations") or [])
        return f"生成答案（{len(ans)} 字符 · {cites} 处引用）"
    return "_(无摘要)_"


def _format_tool_log(log: list[dict]) -> str:
    if not log:
        return "*尚无工具调用记录。提交问题后这里会按节点（planner / executor / verifier / synthesizer）显示每一步的产出摘要。*"
    blocks: list[str] = []
    for entry in log:
        node = entry.get("node", "?")
        ts = entry.get("ts", "")
        summary = entry.get("summary", "")
        label = _NODE_LABELS.get(node, node)
        blocks.append(f"#### {label}\n`{ts}` · {summary}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Conversation state helpers
# ---------------------------------------------------------------------------
def _new_conv_id() -> str:
    return f"c_{uuid.uuid4().hex[:8]}"


def _empty_conv() -> dict:
    return {
        "title": "新对话",
        "history": [],
        "trace": {},
        "citations": [],
        "tool_log": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _ensure_active(convs: dict, active_id: str | None) -> tuple[dict, str]:
    convs = dict(convs or {})
    if not active_id or active_id not in convs:
        active_id = _new_conv_id()
        convs[active_id] = _empty_conv()
    return convs, active_id


def _truncate_title(text: str, n: int = 26) -> str:
    text = text.strip().replace("\n", " ")
    return text[:n] + ("…" if len(text) > n else "")


# ---------------------------------------------------------------------------
# Agent driver
# ---------------------------------------------------------------------------
def _initial_state(message: str) -> dict[str, Any]:
    return {
        "query": message,
        "plan_cursor": 0,
        "reflexion_iter": 0,
        "is_sufficient": False,
        "retrieved_pages": [],
        "extracted_facts": [],
        "computed_values": [],
    }


def _merge_delta(state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    for k, v in delta.items():
        if k in _LIST_REDUCED_KEYS and isinstance(v, list):
            merged[k] = list(merged.get(k, [])) + v
        else:
            merged[k] = v
    return merged


def _stream_agent(message: str) -> Iterable[tuple[str, dict[str, Any], str, dict[str, Any]]]:
    state = _initial_state(message)
    for update in _app.stream(state, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for node_name, delta in update.items():
            if not isinstance(delta, dict):
                continue
            state = _merge_delta(state, delta)
            yield _NODE_STATUS.get(node_name, node_name), state, node_name, delta


# ---------------------------------------------------------------------------
# Chat handlers
# ---------------------------------------------------------------------------
# chat_fn output order (must match wiring):
#  0: chatbot, 1: msg_box, 2: convs_state, 3: active_id_state,
#  4: trace_md, 5: cite_strip, 6: tool_log_md, 7: status_md,
#  8: hero_html, 9: chips_row
def chat_fn(message: str, convs: dict, active_id: str | None):
    convs, active_id = _ensure_active(convs, active_id)

    if not message or not message.strip():
        # No-op: surface a hint, leave state untouched.
        conv = convs[active_id]
        history = conv.get("history") or []
        yield (
            history,
            gr.update(),
            convs,
            active_id,
            _render_trace(conv.get("trace") or {}),
            gr.update(),
            _format_tool_log(conv.get("tool_log") or []),
            "_(请输入问题)_",
            gr.update(visible=not history),
            gr.update(visible=not history),
        )
        return

    msg = message.strip()
    conv = dict(convs[active_id])
    base_history = list(conv.get("history") or [])

    # Set title from first user message.
    if not base_history or conv.get("title") in (None, "新对话", ""):
        conv["title"] = _truncate_title(msg)

    pending_history = base_history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": "_正在思考…_"},
    ]
    conv["history"] = pending_history
    convs[active_id] = conv

    # First yield: hide hero/chips, clear input, push placeholder bubble.
    yield (
        pending_history,
        gr.update(value=""),
        convs,
        active_id,
        "_(开始执行)_",
        gr.update(value=[], visible=False),
        _format_tool_log([]),
        "🚀 启动",
        gr.update(visible=False),
        gr.update(visible=False),
    )

    last_state: dict[str, Any] = _initial_state(msg)
    last_status = "🚀 启动"
    tool_log: list[dict] = []

    try:
        for status, accum_state, node, delta in _stream_agent(msg):
            last_status = status
            last_state = accum_state
            tool_log.append({
                "node": node,
                "ts": datetime.now().strftime("%H:%M:%S"),
                "summary": _summarize_node(node, delta),
            })
            yield (
                pending_history,
                gr.update(),
                convs,
                active_id,
                _render_trace(accum_state),
                gr.update(),
                _format_tool_log(tool_log),
                status,
                gr.update(visible=False),
                gr.update(visible=False),
            )
    except Exception as e:
        err = f"❌ 执行失败: {e}"
        final_history = base_history + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": err},
        ]
        conv["history"] = final_history
        conv["tool_log"] = tool_log
        convs[active_id] = conv
        yield (
            final_history,
            gr.update(value=""),
            convs,
            active_id,
            _render_trace(last_state),
            gr.update(value=[], visible=False),
            _format_tool_log(tool_log),
            err,
            gr.update(visible=False),
            gr.update(visible=False),
        )
        return

    answer = last_state.get("answer") or "_(no answer)_"
    citations: list[Citation] = last_state.get("citations") or []
    gallery = _gallery_items(citations)

    final_history = base_history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": answer + _format_citations_inline(citations)},
    ]
    conv["history"] = final_history
    conv["trace"] = last_state
    conv["citations"] = citations
    conv["tool_log"] = tool_log
    convs[active_id] = conv

    yield (
        final_history,
        gr.update(value=""),
        convs,
        active_id,
        _render_trace(last_state),
        gr.update(value=gallery, visible=bool(gallery)),
        _format_tool_log(tool_log),
        f"✅ 完成 · {last_status}",
        gr.update(visible=False),
        gr.update(visible=False),
    )


# Switch / new / delete output order (8 outputs):
#  convs_state, active_id_state, chatbot, trace_md, cite_strip,
#  tool_log_md, hero_html, chips_row
def _conv_snapshot_outputs(convs: dict, active_id: str):
    conv = convs[active_id]
    history = conv.get("history") or []
    citations = conv.get("citations") or []
    has_msgs = bool(history)
    return (
        convs,
        active_id,
        history,
        _render_trace(conv.get("trace") or {}),
        gr.update(value=_gallery_items(citations), visible=bool(citations)),
        _format_tool_log(conv.get("tool_log") or []),
        gr.update(visible=not has_msgs),
        gr.update(visible=not has_msgs),
    )


def new_conv_fn(convs: dict):
    convs = dict(convs or {})
    new_id = _new_conv_id()
    convs[new_id] = _empty_conv()
    return _conv_snapshot_outputs(convs, new_id)


def switch_conv_fn(target_id: str, convs: dict):
    convs = dict(convs or {})
    if target_id not in convs:
        return new_conv_fn(convs)
    return _conv_snapshot_outputs(convs, target_id)


def delete_conv_fn(target_id: str, convs: dict, active_id: str):
    convs = dict(convs or {})
    convs.pop(target_id, None)
    if active_id == target_id or active_id not in convs:
        if convs:
            sorted_ids = sorted(
                convs.keys(),
                key=lambda c: convs[c].get("created_at", ""),
                reverse=True,
            )
            active_id = sorted_ids[0]
        else:
            active_id = _new_conv_id()
            convs[active_id] = _empty_conv()
    return _conv_snapshot_outputs(convs, active_id)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="FinDoc Agent",
        fill_height=True,
    ) as demo:
        # ---- Per-session state ----
        # First-time visitors start with one empty conversation so the
        # left rail isn't blank.
        bootstrap_id = _new_conv_id()
        bootstrap_convs = {bootstrap_id: _empty_conv()}
        conversations_state = gr.State(bootstrap_convs)
        active_id_state = gr.State(bootstrap_id)

        # ---- Header ----
        with gr.Row(elem_id="app-header"):
            gr.HTML(
                '<h1>FinDoc Agent</h1>'
                '<span class="subtitle">视觉密集型金融文档 · 多模态检索增强</span>'
            )

        # ---- Body: 3 columns ----
        with gr.Row(elem_id="body-row"):
            # ===== Left: conversation history =====
            with gr.Column(scale=1, min_width=240, elem_id="left-sidebar"):
                new_chat_btn = gr.Button(
                    "＋ 新建对话",
                    elem_id="new-chat-btn",
                    variant="primary",
                )
                gr.HTML('<div class="section-title">历史对话</div>')

                # Dynamic conversation list. Re-renders whenever the
                # convs dict or active id changes.
                @gr.render(inputs=[conversations_state, active_id_state])
                def _render_conv_list(convs: dict, active_id: str):
                    if not convs:
                        gr.HTML(
                            '<div class="empty-hint">暂无历史对话</div>',
                            elem_id="conv-list",
                        )
                        return
                    sorted_items = sorted(
                        convs.items(),
                        key=lambda kv: kv[1].get("created_at", ""),
                        reverse=True,
                    )
                    for cid, conv in sorted_items:
                        is_active = cid == active_id
                        row_class = "conv-active" if is_active else "conv-inactive"
                        with gr.Row(elem_classes=["conv-row", row_class]):
                            sw_btn = gr.Button(
                                conv.get("title") or "新对话",
                                elem_classes=["conv-btn"],
                                size="sm",
                            )
                            del_btn = gr.Button(
                                "×",
                                elem_classes=["conv-del"],
                                size="sm",
                            )
                        sw_btn.click(
                            fn=switch_conv_fn,
                            inputs=[gr.State(cid), conversations_state],
                            outputs=[
                                conversations_state, active_id_state,
                                chatbot, trace_md, cite_strip,
                                tool_log_md, hero_html, chips_row,
                            ],
                        )
                        del_btn.click(
                            fn=delete_conv_fn,
                            inputs=[gr.State(cid), conversations_state, active_id_state],
                            outputs=[
                                conversations_state, active_id_state,
                                chatbot, trace_md, cite_strip,
                                tool_log_md, hero_html, chips_row,
                            ],
                        )

            # ===== Center: chat =====
            with gr.Column(scale=4, elem_id="chat-col"):
                with gr.Column(elem_id="chat-inner"):
                    hero_html = gr.HTML(
                        '<div id="hero-html">'
                        '<span class="badge">FinDoc Agent · Beta</span>'
                        '<h2>问我关于<span class="y">年报</span>的任何事</h2>'
                        '<p>我会规划检索路径、读取原页内容，必要时计算并给出带 '
                        '<code>[doc p.X]</code> 引用的答案。'
                        '右侧面板会同步显示每一步的工具调用与原页缩略图。</p>'
                        '</div>',
                        visible=True,
                    )

                    with gr.Row(elem_id="chips-grid", visible=True) as chips_row:
                        chip_buttons = [
                            gr.Button(text, variant="secondary", size="md")
                            for text in _SUGGESTIONS
                        ]

                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        height=520,
                        show_label=False,
                        layout="bubble",
                        avatar_images=(None, None),
                        buttons=["copy"],
                    )

                    with gr.Column(elem_id="input-wrap"):
                        msg_box = gr.Textbox(
                            placeholder="向 FinDoc Agent 提问，按 Enter 发送…",
                            show_label=False,
                            lines=1,
                            max_lines=6,
                            submit_btn=True,
                            elem_id="msg-box",
                            container=False,
                        )
                        status_md = gr.Markdown("", elem_id="status-line")

            # ===== Right: multi-function panel =====
            with gr.Column(scale=2, min_width=300, elem_id="right-sidebar"):
                with gr.Tabs():
                    with gr.TabItem("🧭 执行轨迹"):
                        trace_md = gr.Markdown(
                            "_(尚无轨迹)_",
                            elem_id="right-trace-md",
                        )
                    with gr.TabItem("🔧 工具调用"):
                        tool_log_md = gr.Markdown(
                            _format_tool_log([]),
                            elem_id="right-tools-md",
                        )
                    with gr.TabItem("🖼️ 引用页"):
                        cite_strip = gr.Gallery(
                            elem_id="right-cite-strip",
                            columns=2,
                            height=420,
                            show_label=False,
                            object_fit="cover",
                            visible=False,
                            allow_preview=True,
                        )
                        cite_empty_html = gr.HTML(
                            '<div id="right-cite-empty">'
                            '提交问题后，被引用的原页缩略图会出现在这里。'
                            '</div>'
                        )
                    with gr.TabItem("📚 文档库"):
                        doc_md = gr.Markdown(
                            _doc_summary_md(),
                            elem_id="right-doc-md",
                        )
                        refresh_btn = gr.Button(
                            "🔄 刷新文档库",
                            elem_id="refresh-doc-btn",
                            size="sm",
                        )

        # ---- Wiring ----
        chat_outputs = [
            chatbot, msg_box, conversations_state, active_id_state,
            trace_md, cite_strip, tool_log_md, status_md,
            hero_html, chips_row,
        ]
        msg_box.submit(
            chat_fn,
            inputs=[msg_box, conversations_state, active_id_state],
            outputs=chat_outputs,
        )

        for btn, text in zip(chip_buttons, _SUGGESTIONS):
            btn.click(
                fn=lambda t=text: t,
                outputs=msg_box,
            ).then(
                chat_fn,
                inputs=[msg_box, conversations_state, active_id_state],
                outputs=chat_outputs,
            )

        new_chat_outputs = [
            conversations_state, active_id_state,
            chatbot, trace_md, cite_strip,
            tool_log_md, hero_html, chips_row,
        ]
        new_chat_btn.click(
            new_conv_fn,
            inputs=[conversations_state],
            outputs=new_chat_outputs,
        )

        refresh_btn.click(_doc_summary_md, outputs=doc_md)

    return demo


def main() -> None:
    build_ui().launch(
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.amber,
            secondary_hue=gr.themes.colors.yellow,
            neutral_hue=gr.themes.colors.stone,
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
        ),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
