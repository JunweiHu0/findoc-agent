"""Chainlit frontend for FinDoc Agent — minimal, LangGraph-native.

Run:
    chainlit run app/chainlit_app.py -w

Why Chainlit over the Gradio app (see LEARNLOG §"前端选型"):
    - LangchainCallbackHandler renders每个 LangGraph 节点为可折叠 Step 自动
    - cl.Image 元素天然展示召回页缩略图（点击放大）
    - 后端零改动：`agent/`、`tools/`、`ingestion/` 完全不动

This file replaces the chat plumbing in `gradio_app.py`. The Gradio app
is kept as a fallback while Chainlit is being polished.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Chainlit's `load_module` imports this file by spec without injecting the
# project root into sys.path — unlike `python -m app.chainlit_app`. We bootstrap
# it manually so `from agent.* import ...` resolves regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chainlit as cl
from loguru import logger

from agent.config import INDEX_DIR, PAGES_DIR
from agent.graph import compile_graph
from agent.state import Citation, PageHit


# ---------------------------------------------------------------------------
# Graph: compile once at import. Stateless across sessions; safe to share.
# (Per-session mutable state lives entirely in AgentState passed to invoke.)
# ---------------------------------------------------------------------------
_GRAPH = compile_graph()


# Suppress noisy internal langgraph runnables; surface only the four agent
# nodes (planner / executor / verifier / synthesizer) plus their LLM calls.
_CALLBACK_IGNORE = [
    "ChannelRead",
    "ChannelWrite",
    "RunnableLambda",
    "RunnableSequence",
    "RunnableParallel",
    "RunnableAssign",
    "LangGraph",
]


# ---------------------------------------------------------------------------
# Doc memory helpers (mirrored from gradio_app for self-containment)
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


def _resolve_page_image(doc_id: str, page_num: int) -> str | None:
    candidate = PAGES_DIR / doc_id / f"p{page_num:03d}.png"
    return str(candidate) if candidate.exists() else None


def _initial_state(query: str) -> dict[str, Any]:
    return {
        "query": query,
        "plan_cursor": 0,
        "reflexion_iter": 0,
        "is_sufficient": False,
        "retrieved_pages": [],
        "extracted_facts": [],
        "computed_values": [],
    }


def _format_citations(citations: list[Citation]) -> str:
    if not citations:
        return ""
    refs = " ".join(f"`[{c.doc_id} p.{c.page_num}]`" for c in citations)
    return f"\n\n**引用**：{refs}"


def _page_elements(pages: list[PageHit]) -> list[cl.Image]:
    """De-dup retrieved pages and resolve to inline cl.Image elements."""
    seen: set[tuple[str, int]] = set()
    elements: list[cl.Image] = []
    for hit in pages or []:
        key = (hit.doc_id, hit.page_num)
        if key in seen:
            continue
        seen.add(key)
        path = _resolve_page_image(hit.doc_id, hit.page_num)
        if path:
            elements.append(
                cl.Image(
                    path=path,
                    name=f"{hit.doc_id} p.{hit.page_num}",
                    display="inline",
                )
            )
    return elements


# ---------------------------------------------------------------------------
# Chainlit lifecycle
# ---------------------------------------------------------------------------
@cl.set_starters
async def starters():
    return [
        cl.Starter(
            label="📊 茅台 2023 年营收",
            message="贵州茅台 2023 年的营业收入是多少？",
        ),
        cl.Starter(
            label="🆚 毛利率对比",
            message="对比贵州茅台、宁德时代 2023 年的毛利率",
        ),
        cl.Starter(
            label="👥 招行员工数",
            message="招商银行 2024 年末员工总数是多少？",
        ),
        cl.Starter(
            label="🧪 恒瑞研发占比",
            message="恒瑞医药 2024 年研发投入占营业收入的比例",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    docs = _load_doc_memory()
    if not docs:
        await cl.Message(
            content=(
                "⚠️ **尚未建立索引**。请先运行：\n\n"
                "```bash\npython -m ingestion.build_index\n```"
            ),
            author="system",
        ).send()
        return

    body = "**已索引文档**（共 {n} 份）\n\n".format(n=len(docs))
    body += "\n".join(f"- `{d['doc_id']}` · {d['page_count']} 页" for d in docs)
    await cl.Message(content=body, author="system").send()


@cl.on_message
async def on_message(msg: cl.Message):
    query = (msg.content or "").strip()
    if not query:
        await cl.Message(content="_(请输入问题)_").send()
        return

    cb = cl.LangchainCallbackHandler(
        stream_final_answer=False,
        to_ignore=_CALLBACK_IGNORE,
    )
    state = _initial_state(query)

    try:
        final = await _GRAPH.ainvoke(state, config={"callbacks": [cb]})
    except Exception as e:
        logger.exception("graph invocation failed")
        await cl.Message(content=f"❌ Agent 执行失败：`{type(e).__name__}: {e}`").send()
        return

    answer = (final.get("answer") or "_(空回答)_").strip()
    citations = final.get("citations") or []
    answer += _format_citations(citations)

    elements = _page_elements(final.get("retrieved_pages") or [])
    await cl.Message(content=answer, elements=elements).send()
