"""Chainlit frontend for FinDoc Agent — pure presentation layer.

Run:
    chainlit run app/chainlit_app.py -w

Requires the FastAPI backend running (default http://localhost:8001).
All agent logic, retrieval, and LLM/VLM calls happen in the backend.
This file only handles UI rendering: Steps, messages, images, citations.
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
import httpx
from loguru import logger

from agent.config import CONFIG, PAGES_DIR

# Backend URL — configurable via config.yaml
_BACKEND_URL = CONFIG.get("backend", {}).get("url", "http://localhost:8001")

_NODE_LABELS: dict[str, str] = {
    "planner": "\U0001f9e0 Planner",
    "executor": "\U0001f527 Executor",
    "verifier": "\U0001f50d Verifier",
    "synthesizer": "✍️ Synthesizer",
}


# ---------------------------------------------------------------------------
# Helpers (UI-only — no agent imports)
# ---------------------------------------------------------------------------
def _truncate(text: str, n: int = 35) -> str:
    text = text.strip().replace("\n", " ")
    return text[:n] + ("…" if len(text) > n else "")


def _resolve_page_image(doc_id: str, page_num: int) -> str | None:
    candidate = PAGES_DIR / doc_id / f"p{page_num:03d}.png"
    return str(candidate) if candidate.exists() else None


def _format_citations(citations: list[dict]) -> str:
    if not citations:
        return ""
    refs = " ".join(f"`[{c['doc_id']} p.{c['page_num']}]`" for c in citations)
    return f"\n\n**引用**：{refs}"


def _page_elements(pages: list[dict]) -> list[cl.Image]:
    seen: set[tuple[str, int]] = set()
    elements: list[cl.Image] = []
    for hit in pages or []:
        key = (hit.get("doc_id", ""), hit.get("page_num", 0))
        if key in seen:
            continue
        seen.add(key)
        path = _resolve_page_image(hit["doc_id"], hit["page_num"])
        if path:
            elements.append(
                cl.Image(
                    path=path,
                    name=f"{hit['doc_id']} p.{hit['page_num']}",
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
        cl.Starter(label="\U0001f4ca 茅台 2023 年营收", message="贵州茅台2023年的营业收入是多少？"),
        cl.Starter(label="\U0001f53c 毛利率对比", message="对比贵州茅台、宁德时代 2023 年的毛利率"),
        cl.Starter(label="\U0001f465 招行员工数", message="招商银行 2024 年末员工总数是多少？"),
        cl.Starter(label="\U0001f9ea 恒瑞研发占比", message="恒瑞医药 2024 年研发投入占营业收入的比例"),
    ]


@cl.on_chat_start
async def on_chat_start():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{_BACKEND_URL}/api/v1/docs", timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        await cl.Message(
            content=f"⚠️ **无法连接后端** `{_BACKEND_URL}`\n\n请先启动：\n```bash\nPYTHONPATH=. uvicorn backend.server:app --port 8001\n```",
            author="system",
        ).send()
        return

    docs = data.get("docs") or []
    if not docs:
        await cl.Message(
            content="⚠️ **尚未建立索引**。请先运行：\n```bash\npython -m ingestion.build_index\n```",
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

    _status_step: cl.Step | None = None
    final_answer = ""
    final_citations: list[dict] = []
    final_pages: list[dict] = []

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{_BACKEND_URL}/api/v1/query",
                json={"query": query},
                timeout=300.0,
            ) as resp:
                if resp.status_code != 200:
                    err_text = await resp.aread()
                    await cl.Message(content=f"❌ 后端返回 {resp.status_code}: {err_text.decode()}").send()
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if not payload_str:
                        continue

                    try:
                        data = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue

                    etype = data.get("type", "")

                    if etype == "error":
                        await cl.Message(content=f"❌ Agent 执行失败：{data.get('message', '未知错误')}").send()
                        return

                    if etype == "status":
                        # Inline progress — update the running status step
                        msg_text = data.get("message", "")
                        if _status_step is None:
                            _status_step = cl.Step(name=f"⏳ {msg_text}", type="tool")
                            await _status_step.send()
                        else:
                            _status_step.name = f"⏳ {msg_text}"
                            await _status_step.update()
                        continue

                    if etype == "done":
                        # Remove the status step when agent completes
                        if _status_step is not None:
                            _status_step.name = "✅ 完成"
                            await _status_step.update()
                        final_answer = data.get("answer", "")
                        final_citations = data.get("citations", [])
                        final_pages = data.get("retrieved_pages", [])
                        break

                    # Node update
                    node = data.get("node", "")
                    summary = data.get("summary", "")
                    content = data.get("content", "")

                    label = _NODE_LABELS.get(node, node)
                    step_name = f"{label} · {_truncate(summary, 35)}"

                    async with cl.Step(name=step_name, type="tool") as step:
                        step.output = content

    except httpx.ConnectError:
        await cl.Message(
            content=f"❌ **无法连接后端** `{_BACKEND_URL}`\n\n请确认后端已启动：\n```bash\nPYTHONPATH=. uvicorn backend.server:app --port 8001\n```"
        ).send()
        return
    except Exception as e:
        logger.exception("frontend streaming failed")
        await cl.Message(content=f"❌ 连接后端异常：`{type(e).__name__}: {e}`").send()
        return

    # Final answer message
    answer = final_answer or "*(空回答)*"
    answer += _format_citations(final_citations)
    elements = _page_elements(final_pages)
    await cl.Message(content=answer, elements=elements).send()
