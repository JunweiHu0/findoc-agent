"""Chainlit frontend for FinDoc Agent — P9: detailed workflow visibility.

Run:
    chainlit run app/chainlit_app.py -w

Each LangGraph node creates a cl.Step with a descriptive name (30-char truncated
summary of what the node produced) and full delta content shown when expanded.
The auto-rendering is preserved through manual Step creation in the astream loop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chainlit as cl
from loguru import logger

from agent.config import INDEX_DIR, PAGES_DIR
from agent.graph import compile_graph
from agent.state import Citation, PageHit

_GRAPH = compile_graph()

_NODE_LABELS: dict[str, str] = {
    "planner": "\U0001f9e0 Planner",
    "executor": "\U0001f527 Executor",
    "verifier": "\U0001f50d Verifier",
    "synthesizer": "✍️ Synthesizer",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _summarize_node(node: str, delta: dict[str, Any]) -> str:
    """Plain-text one-liner summarising what a node produced.  Used in Step names."""
    if node == "planner":
        plan = delta.get("plan") or []
        if not plan:
            return "未生成计划"
        first = plan[0].sub_query if hasattr(plan[0], "sub_query") else str(plan[0])
        return f"产出{len(plan)}步计划: {first}"
    if node == "executor":
        pages = len(delta.get("retrieved_pages") or [])
        facts = len(delta.get("extracted_facts") or [])
        cvs = len(delta.get("computed_values") or [])
        return f"检索{pages}页 · 抽取{facts}事实 · 计算{cvs}"
    if node == "verifier":
        return "✅ 充分" if delta.get("is_sufficient") else "↻ 触发再检索"
    if node == "synthesizer":
        ans = delta.get("answer") or ""
        cites = len(delta.get("citations") or [])
        return f"生成答案({len(ans)}字符 · {cites}引用)"
    return "完成"


def _format_delta(node: str, delta: dict[str, Any]) -> str:
    """Render the complete node delta as structured markdown for the Step body."""
    if node == "planner":
        plan = delta.get("plan") or []
        if not plan:
            return "*(空计划)*"
        lines = [f"**执行计划** — {len(plan)} 步\n"]
        for i, task in enumerate(plan):
            sq = task.sub_query if hasattr(task, "sub_query") else str(task)
            td = getattr(task, "target_doc", None)
            target = f" `[{td}]`" if td else ""
            es = getattr(task, "expected_output_schema", "text")
            lines.append(f"{i+1}. `{sq}`{target}  → *{es}*")
        return "\n".join(lines)

    if node == "executor":
        parts: list[str] = []
        pages: list[PageHit] = delta.get("retrieved_pages") or []
        if pages:
            parts.append(f"**检索页面** ({len(pages)})\n")
            for p in pages:
                parts.append(f"- `{p.doc_id}` p.{p.page_num:03d}  score={p.score:.4f}")
        facts = delta.get("extracted_facts") or []
        if facts:
            parts.append(f"\n**抽取事实** ({len(facts)})\n")
            for f in facts:
                src = getattr(f, "source_doc", "?")
                sp = getattr(f, "source_page", "?")
                parts.append(f"- [{src} p.{sp}] {f.text}")
        values = delta.get("computed_values") or []
        if values:
            parts.append(f"\n**计算结果** ({len(values)})\n")
            for v in values:
                parts.append(f"- `{v.expr}` = **{v.value}**")
        return "\n".join(parts) if parts else "*(无输出)*"

    if node == "verifier":
        suff = "✅ 信息充分" if delta.get("is_sufficient") else "❌ 信息不充分"
        miss = delta.get("missing_info") or ""
        out = f"**判断**: {suff}"
        if miss:
            out += f"\n\n**缺失信息**: {miss}"
        reflexion = delta.get("reflexion_iter")
        if reflexion is not None:
            out += f"\n\n轮次: {reflexion}"
        return out

    if node == "synthesizer":
        ans = delta.get("answer") or ""
        cites = delta.get("citations") or []
        out = f"**生成答案** — {len(ans)} 字符 · {len(cites)} 处引用\n\n---\n\n{ans}"
        if cites:
            out += "\n\n---\n\n**引用**: " + " ".join(
                f"`[{c.doc_id} p.{c.page_num}]`" for c in cites
            )
        return out

    return f"```json\n{json.dumps(delta, ensure_ascii=False, indent=2, default=str)}\n```"


def _truncate(text: str, n: int = 35) -> str:
    text = text.strip().replace("\n", " ")
    return text[:n] + ("…" if len(text) > n else "")


# ---------------------------------------------------------------------------
# Doc memory & page helpers
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
            label="\U0001f4ca 茅台 2023 年营收",
            message="贵州茅台2023年的营业收入是多少？",
        ),
        cl.Starter(
            label="\U0001f53c 毛利率对比",
            message="对比贵州茅台、宁德时代 2023 年的毛利率",
        ),
        cl.Starter(
            label="\U0001f465 招行员工数",
            message="招商银行 2024 年末员工总数是多少？",
        ),
        cl.Starter(
            label="\U0001f9ea 恒瑞研发占比",
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
        await cl.Message(content="*(请输入问题)*").send()
        return

    state = _initial_state(query)
    accumulated: dict[str, Any] = dict(state)

    # ---- stream LangGraph nodes as individually collapsible Steps ----
    try:
        async for chunk in _GRAPH.astream(state, stream_mode="updates"):
            for node_name, delta in chunk.items():
                summary = _summarize_node(node_name, delta)
                label = _NODE_LABELS.get(node_name, node_name)
                step_name = f"{label} · {_truncate(summary, 35)}"

                async with cl.Step(name=step_name, type="tool") as step:
                    step.output = _format_delta(node_name, delta)

                # Merge delta into accumulated state
                for key, value in delta.items():
                    if key in accumulated and isinstance(accumulated[key], list) and isinstance(value, list):
                        accumulated[key] = accumulated[key] + value
                    else:
                        accumulated[key] = value

    except Exception as e:
        logger.exception("graph streaming failed")
        await cl.Message(content=f"❌ Agent 执行失败：`{type(e).__name__}: {e}`").send()
        return

    # ---- final answer message ----
    answer = (accumulated.get("answer") or "*(空回答)*").strip()
    citations = accumulated.get("citations") or []
    answer += _format_citations(citations)

    elements = _page_elements(accumulated.get("retrieved_pages") or [])
    await cl.Message(content=answer, elements=elements).send()
