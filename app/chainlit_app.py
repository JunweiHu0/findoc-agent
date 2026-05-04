"""Chainlit frontend for FinDoc Agent — pure presentation layer.

Run:
    chainlit run app/chainlit_app.py -w

Requires the FastAPI backend running (default http://localhost:8001).
All agent logic, retrieval, VLM calls, persistence happen in the backend.

P13: left-sidebar conversation history via custom DataLayer bridging to
backend SQLite (see app/data_layer.py).

P14: "+ attach" button on the left of the chat input (Chainlit
spontaneous_file_upload). Attached PDFs/images upload to the backend,
get encoded with ColQwen2, and become queryable in the same chat turn.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chainlit as cl
import httpx
from loguru import logger

from agent.config import CONFIG, PAGES_DIR
from app.data_layer import FinDocDataLayer

_BACKEND_URL = CONFIG.get("backend", {}).get("url", "http://localhost:8001")

_NODE_LABELS: dict[str, str] = {
    "planner": "\U0001f9e0 Planner",
    "executor": "\U0001f527 Executor",
    "verifier": "\U0001f50d Verifier",
    "synthesizer": "✍️ Synthesizer",
}

_UPLOAD_ACCEPT_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp"}


# ---------------------------------------------------------------------------
# Chainlit data layer + auth (P13: enable threads sidebar)
# ---------------------------------------------------------------------------
@cl.data_layer
def _get_data_layer():
    return FinDocDataLayer()


@cl.header_auth_callback
async def _auth(headers) -> Optional[cl.User]:
    """Single-user mode — accept everyone as 'local'. Required for the
    threads sidebar to render."""
    return cl.User(identifier="local", metadata={"role": "user"})


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
# P14: Inline upload — drains attachments from a Chainlit message and pushes
# each through the backend upload pipeline. Returns the list of doc_ids
# successfully indexed (so they can be referenced in the same turn's query).
# ---------------------------------------------------------------------------
async def _handle_inline_uploads(msg: cl.Message) -> list[str]:
    files = [e for e in (msg.elements or []) if getattr(e, "path", None)]
    if not files:
        return []

    doc_ids: list[str] = []
    for elem in files:
        path_str = getattr(elem, "path", None)
        if not path_str:
            continue
        fpath = Path(path_str)
        suffix = fpath.suffix.lower()
        if suffix not in _UPLOAD_ACCEPT_SUFFIXES:
            await cl.Message(
                content=f"⚠️ 跳过不支持的文件 `{fpath.name}`（仅支持 PDF / 图片）",
                author="system",
            ).send()
            continue

        filename = getattr(elem, "name", fpath.name) or fpath.name
        mime = getattr(elem, "mime", None) or ("application/pdf" if suffix == ".pdf" else f"image/{suffix.lstrip('.')}")

        upload_step = cl.Step(name=f"📤 上传 {filename}", type="tool")
        await upload_step.send()

        try:
            doc_id = await _upload_and_track(fpath, filename, mime, upload_step)
            if doc_id:
                doc_ids.append(doc_id)
        except Exception as e:
            logger.exception("inline upload failed")
            upload_step.name = f"❌ {filename} 上传失败"
            upload_step.output = f"{type(e).__name__}: {e}"
            await upload_step.update()

    return doc_ids


async def _upload_and_track(
    fpath: Path,
    filename: str,
    mime: str,
    upload_step: cl.Step,
) -> str | None:
    """POST file to backend then stream SSE progress; mutate upload_step."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        with open(fpath, "rb") as fh:
            resp = await client.post(
                f"{_BACKEND_URL}/api/v1/upload",
                files={"file": (filename, fh, mime)},
            )
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            upload_step.name = f"❌ {filename} 上传失败"
            upload_step.output = str(detail)
            await upload_step.update()
            return None

        result = resp.json()
        upload_id = result["upload_id"]
        doc_id = result["doc_id"]

        upload_step.name = f"⏳ {filename} → `{doc_id}` 排队中"
        await upload_step.update()

        async with client.stream(
            "GET",
            f"{_BACKEND_URL}/api/v1/upload/{upload_id}/status",
            timeout=600.0,
        ) as status_resp:
            async for line in status_resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if not payload_str:
                    continue
                try:
                    info = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                status = info.get("status", "queued")
                stage_msg = info.get("message", "")
                emoji = {
                    "save": "💾", "pages": "🖼️", "encode": "🧠",
                    "index": "📦", "qdrant": "🗄️", "register": "📋",
                    "done": "✅", "failed": "❌", "queued": "⏳",
                    "encoding": "🧠",
                }.get(status, "⏳")

                upload_step.name = f"{emoji} {filename} · {stage_msg}"
                await upload_step.update()

                if status == "done":
                    upload_step.name = f"✅ `{doc_id}` 已索引"
                    upload_step.output = f"文档 `{doc_id}` 已可被检索。"
                    await upload_step.update()
                    return doc_id
                if status == "failed":
                    upload_step.name = f"❌ {filename} 索引失败"
                    upload_step.output = stage_msg or "未知错误"
                    await upload_step.update()
                    return None
    return None


# ---------------------------------------------------------------------------
# Knowledge base admin lives in the right-side panel (public/sidebar.js).
# Inline list/delete/reindex callbacks were removed so the chat stays clean.
# ---------------------------------------------------------------------------


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
    cl.user_session.set("conv_id", None)
    cl.user_session.set("uploaded_doc_ids", [])

    # Silent backend probe — only surface a message if the backend is unreachable.
    # The user's own documents live in the right-side panel (see public/sidebar.js).
    # System-default offline indexes are intentionally hidden from the user.
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{_BACKEND_URL}/api/v1/health", timeout=5.0)
            resp.raise_for_status()
    except Exception:
        await cl.Message(
            content=(
                f"⚠️ **无法连接后端** `{_BACKEND_URL}`\n\n请先启动：\n"
                "```bash\nPYTHONPATH=. uvicorn backend.server:app --port 8001\n```"
            ),
            author="system",
        ).send()


@cl.on_chat_resume
async def on_chat_resume(thread):
    """Restore session state when resuming a thread from the sidebar."""
    cl.user_session.set("conv_id", thread.get("id"))
    cl.user_session.set("uploaded_doc_ids", [])


@cl.on_message
async def on_message(msg: cl.Message):
    # P14 — drain any attached files first, build doc_filter from successful uploads
    new_doc_ids = await _handle_inline_uploads(msg)

    accumulated_doc_ids: list[str] = list(cl.user_session.get("uploaded_doc_ids") or [])
    accumulated_doc_ids.extend(d for d in new_doc_ids if d not in accumulated_doc_ids)
    cl.user_session.set("uploaded_doc_ids", accumulated_doc_ids)

    query = (msg.content or "").strip()
    if not query:
        # User uploaded a file without typing anything — acknowledge and stop
        if new_doc_ids:
            await cl.Message(
                content=f"✅ 已上传 {len(new_doc_ids)} 个文档。请输入你的问题，我会优先在新文档中检索。",
                author="system",
            ).send()
        else:
            await cl.Message(content="*(请输入问题)*").send()
        return

    # Use Chainlit's thread_id as the canonical conversation key — this keeps
    # the threads sidebar (data layer) and backend SQLite in sync without
    # double-persisting messages.
    try:
        conv_id = cl.context.session.thread_id
    except Exception:
        conv_id = cl.user_session.get("conv_id")
    cl.user_session.set("conv_id", conv_id)

    _status_step: cl.Step | None = None
    final_answer = ""
    final_citations: list[dict] = []
    final_pages: list[dict] = []
    streaming_msg: cl.Message | None = None  # P16: token-by-token typewriter
    streamed_text = ""

    body: dict = {"query": query}
    if conv_id:
        body["conv_id"] = conv_id
    if accumulated_doc_ids:
        body["doc_filter"] = accumulated_doc_ids

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{_BACKEND_URL}/api/v1/query",
                json=body,
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
                        msg_text = data.get("message", "")
                        if _status_step is None:
                            _status_step = cl.Step(name=f"⏳ {msg_text}", type="tool")
                            await _status_step.send()
                        else:
                            _status_step.name = f"⏳ {msg_text}"
                            await _status_step.update()
                        continue

                    if etype == "token":
                        tok = data.get("token", "")
                        if not tok:
                            continue
                        if streaming_msg is None:
                            streaming_msg = cl.Message(content="")
                            await streaming_msg.send()
                        streamed_text += tok
                        await streaming_msg.stream_token(tok)
                        continue

                    if etype == "done":
                        if _status_step is not None:
                            _status_step.name = "✅ 完成"
                            await _status_step.update()
                        final_answer = data.get("answer", "")
                        final_citations = data.get("citations", [])
                        final_pages = data.get("retrieved_pages", [])
                        final_grounding_score = data.get("grounding_score", 1.0)
                        new_conv_id = data.get("conv_id")
                        if new_conv_id:
                            cl.user_session.set("conv_id", new_conv_id)
                        break

                    node = data.get("node", "")
                    summary = data.get("summary", "")
                    content = data.get("content", "")

                    label = _NODE_LABELS.get(node, node)
                    step_name = f"{label} · {_truncate(summary, 35)}"

                    async with cl.Step(name=step_name, type="tool") as step:
                        step.output = content

    except httpx.ConnectError:
        await cl.Message(
            content=(
                f"❌ **无法连接后端** `{_BACKEND_URL}`\n\n请确认后端已启动：\n"
                "```bash\nPYTHONPATH=. uvicorn backend.server:app --port 8001\n```"
            )
        ).send()
        return
    except Exception as e:
        logger.exception("frontend streaming failed")
        await cl.Message(content=f"❌ 连接后端异常：`{type(e).__name__}: {e}`").send()
        return

    elements = _page_elements(final_pages)
    citations_md = _format_citations(final_citations)

    if streaming_msg is not None:
        # Tokens already on screen — append citations + attach page elements in place.
        body = (final_answer or streamed_text) + citations_md
        streaming_msg.content = body
        if elements:
            streaming_msg.elements = elements
        await streaming_msg.update()
    else:
        answer = (final_answer or "*(空回答)*") + citations_md
        await cl.Message(content=answer, elements=elements).send()