"""FinDoc Agent — FastAPI backend with SSE streaming.

Start:
    PYTHONPATH=. uvicorn backend.server:app --host 0.0.0.0 --port 8001

Endpoints:
    POST /api/v1/query   — SSE stream of agent node updates + final answer
    GET  /api/v1/docs    — indexed document list
    GET  /api/v1/health  — backend status

The agent/ tools/ ingestion/ directories are completely untouched — this
server just wraps them behind HTTP.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

# Bootstrap project root for agent.* imports
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from agent.config import CONFIG, INDEX_DIR, PAGES_DIR
from agent.graph import compile_graph
from agent.state import Citation, PageHit
from backend.schemas import (
    CitationOut, ConversationCreate, ConversationDetail, ConversationOut,
    ConversationUpdate, DocInfo, DocumentOut, HealthResponse, PageHitOut,
    QueryRequest, UploadResponse, UploadStatusOut,
)
from backend import storage

# ---------------------------------------------------------------------------
# Graph — module-level, stateless, safe to share across requests
# ---------------------------------------------------------------------------
_GRAPH = compile_graph()

# ---------------------------------------------------------------------------
# Node summary helpers (moved here from chainlit_app.py)
# ---------------------------------------------------------------------------
def _summarize_node(node: str, delta: dict[str, Any]) -> str:
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
        pages: list = delta.get("retrieved_pages") or []
        if pages:
            parts.append(f"**检索页面** ({len(pages)})\n")
            for p in pages:
                did = getattr(p, "doc_id", "?")
                pn = getattr(p, "page_num", 0)
                sc = getattr(p, "score", 0.0)
                parts.append(f"- `{did}` p.{pn:03d}  score={sc:.4f}")
        facts = delta.get("extracted_facts") or []
        if facts:
            parts.append(f"\n**抽取事实** ({len(facts)})\n")
            for f in facts:
                src = getattr(f, "source_doc", "?")
                sp = getattr(f, "source_page", "?")
                parts.append(f"- [{src} p.{sp}] {getattr(f, 'text', str(f))}")
        values = delta.get("computed_values") or []
        if values:
            parts.append(f"\n**计算结果** ({len(values)})\n")
            for v in values:
                parts.append(f"- `{getattr(v, 'expr', '?')}` = **{getattr(v, 'value', float('nan'))}**")
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
                f"`[{getattr(c, 'doc_id', '?')} p.{getattr(c, 'page_num', '?')}]`" for c in cites
            )
        return out

    return f"```json\n{json.dumps(delta, ensure_ascii=False, indent=2, default=str)}\n```"


# ---------------------------------------------------------------------------
# Agent state helpers
# ---------------------------------------------------------------------------
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


def _page_hit_to_out(hit: PageHit) -> PageHitOut:
    return PageHitOut(
        doc_id=hit.doc_id,
        page_num=hit.page_num,
        score=hit.score,
        image_path=hit.image_path,
    )


def _citation_to_out(c: Citation) -> CitationOut:
    return CitationOut(doc_id=c.doc_id, page_num=c.page_num)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="FinDoc Agent API", version="0.1.0")


@app.on_event("startup")
async def startup():
    """Preload ColQwen2 model + indexes so first request has zero cold-start latency."""
    import asyncio

    colqwen_url = CONFIG.get("services", {}).get("colqwen_url", "")
    if colqwen_url:
        logger.info(f"Remote ColQwen Service configured at {colqwen_url}, skipping local model preload")
        return

    from tools.colpali_tool import preload

    logger.info("Preloading ColQwen2 model and indexes (~45s on first run, ~2s on warm disk cache) ...")
    await asyncio.to_thread(preload)
    logger.info("ColQwen2 model warm — ready to serve")


@app.post("/api/v1/query")
async def query(req: QueryRequest):
    """Run the LangGraph agent in a background thread; stream node + progress events as SSE."""
    import asyncio
    import queue
    import threading

    # Resolve conversation: prefer caller-supplied conv_id (Chainlit thread_id);
    # fall back to a fresh one. Auto-create the row if missing.
    conv_id = req.conv_id
    title = req.query.strip().replace("\n", " ")[:26]
    if not conv_id:
        conv = storage.create_conversation(title=title)
        conv_id = conv["id"]
    else:
        existing = storage.get_conversation(conv_id)
        if not existing:
            storage.create_conversation_with_id(conv_id, title=title)

    node_queue: queue.Queue[tuple[str, dict]] = queue.Queue()
    progress_queue: queue.Queue[str] = queue.Queue()

    def _on_progress(msg: str) -> None:
        progress_queue.put(msg)

    def _run_agent_sync(state: dict) -> None:
        """Execute agent graph synchronously in a background thread."""
        from tools.colpali_tool import set_progress_hook as set_colpali_hook
        from tools.vlm_tool import set_progress_hook as set_vlm_hook
        set_colpali_hook(_on_progress)
        set_vlm_hook(_on_progress)

        accumulated = dict(state)
        try:
            for chunk in _GRAPH.stream(state, stream_mode="updates"):
                for node_name, delta in chunk.items():
                    node_queue.put((node_name, delta))
                    for key, value in delta.items():
                        if key in accumulated and isinstance(accumulated[key], list) and isinstance(value, list):
                            accumulated[key] = accumulated[key] + value
                        else:
                            accumulated[key] = value
            node_queue.put(("__done__", accumulated))
        except Exception as e:
            logger.exception("agent execution failed")
            node_queue.put(("__error__", {"message": f"{type(e).__name__}: {e}"}))

    async def event_stream():
        state = _initial_state(req.query)
        thread = threading.Thread(target=_run_agent_sync, args=(state,), daemon=True)
        thread.start()

        while True:
            # Drain progress queue first — these may arrive during long tool ops
            had_progress = False
            while True:
                try:
                    msg = progress_queue.get_nowait()
                    payload = json.dumps({"type": "status", "message": msg}, ensure_ascii=False)
                    yield f"event: status\ndata: {payload}\n\n"
                    had_progress = True
                except queue.Empty:
                    break

            # Drain node results
            try:
                node_name, data = node_queue.get_nowait()
            except queue.Empty:
                if not had_progress:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.3)
                continue

            if node_name == "__done__":
                accumulated = data
                break
            elif node_name == "__error__":
                err = json.dumps({"type": "error", "message": data["message"]}, ensure_ascii=False)
                yield f"event: error\ndata: {err}\n\n"
                thread.join()
                return

            # Normal node event
            summary = _summarize_node(node_name, data)
            content = _format_delta(node_name, data)
            payload = json.dumps({
                "node": node_name, "summary": summary, "content": content,
            }, ensure_ascii=False)
            yield f"event: node\ndata: {payload}\n\n"

        thread.join()

        accumulated: dict[str, Any] = data  # data from __done__
        answer = (accumulated.get("answer") or "").strip()
        citations = accumulated.get("citations") or []
        pages = accumulated.get("retrieved_pages") or []

        # Persist to conversation history
        try:
            storage.add_message(conv_id, "user", req.query)
            storage.add_message(
                conv_id, "assistant", answer,
                citations=[_citation_to_out(c).model_dump() for c in citations],
                pages=[_page_hit_to_out(p).model_dump() for p in pages],
            )
        except Exception as e:
            logger.warning(f"Failed to persist messages for conv {conv_id}: {e}")

        done = json.dumps({
            "type": "done",
            "conv_id": conv_id,
            "answer": answer,
            "citations": [_citation_to_out(c).model_dump() for c in citations],
            "retrieved_pages": [_page_hit_to_out(p).model_dump() for p in pages],
        }, ensure_ascii=False)
        yield f"event: done\ndata: {done}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/v1/docs", response_model=dict[str, list[DocInfo]])
async def list_docs():
    """Return indexed document list."""
    mem_path = INDEX_DIR / "doc_memory.json"
    if not mem_path.exists():
        return {"docs": []}
    data = json.loads(mem_path.read_text(encoding="utf-8"))
    return {
        "docs": [DocInfo(doc_id=d["doc_id"], page_count=d["page_count"]) for d in data.get("docs", [])]
    }


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """Backend health + basic status."""
    mem_path = INDEX_DIR / "doc_memory.json"
    docs_count = 0
    if mem_path.exists():
        data = json.loads(mem_path.read_text(encoding="utf-8"))
        docs_count = len(data.get("docs", []))
    backend = CONFIG["retriever"].get("backend", "in_memory")
    return HealthResponse(status="ok", docs_count=docs_count, backend=backend)


# ---------------------------------------------------------------------------
# P13: Conversation endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/conversations", response_model=list[ConversationOut])
async def list_conversations():
    return storage.list_conversations()


@app.post("/api/v1/conversations", response_model=ConversationOut)
async def create_conversation(body: ConversationCreate):
    conv = storage.create_conversation(title=body.title)
    return ConversationOut(**conv)


@app.get("/api/v1/conversations/{conv_id}", response_model=ConversationDetail)
async def get_conversation(conv_id: str):
    from fastapi.responses import JSONResponse
    conv = storage.get_conversation(conv_id)
    if not conv:
        return JSONResponse(status_code=404, content={"detail": "Conversation not found"})
    return ConversationDetail(**conv)


@app.patch("/api/v1/conversations/{conv_id}", response_model=ConversationOut)
async def update_conversation(conv_id: str, body: ConversationUpdate):
    from fastapi.responses import JSONResponse
    ok = storage.update_conversation_title(conv_id, body.title)
    if not ok:
        return JSONResponse(status_code=404, content={"detail": "Conversation not found"})
    conv = storage.get_conversation(conv_id)
    return ConversationOut(**conv)


@app.delete("/api/v1/conversations/{conv_id}")
async def delete_conversation_endpoint(conv_id: str):
    from fastapi.responses import JSONResponse
    ok = storage.delete_conversation(conv_id)
    if not ok:
        return JSONResponse(status_code=404, content={"detail": "Conversation not found"})
    return {"deleted": conv_id}


# ---------------------------------------------------------------------------
# P14: Document / Upload endpoints (stubs — pipeline in ingestion/upload.py)
# ---------------------------------------------------------------------------

@app.get("/api/v1/documents", response_model=list[DocumentOut])
async def list_documents_all():
    """List all documents (indexed + uploaded)."""
    docs = storage.list_documents()
    return [DocumentOut(**d) for d in docs]


@app.delete("/api/v1/documents/{doc_id}")
async def delete_doc(doc_id: str):
    """Delete a document: pages + index + Qdrant points + DB record."""
    from fastapi.responses import JSONResponse
    import shutil

    # Remove pages
    pages_dir = PAGES_DIR / doc_id
    if pages_dir.exists():
        shutil.rmtree(pages_dir, ignore_errors=True)

    # Remove index
    index_dir = INDEX_DIR / doc_id
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)

    # Remove from Qdrant
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        qdrant_cfg = CONFIG["retriever"].get("qdrant", {})
        client = QdrantClient(url=qdrant_cfg.get("url", "http://localhost:6333"))
        collection = qdrant_cfg.get("collection_name", "findoc_pages")
        client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )
        logger.info(f"Deleted Qdrant points for {doc_id}")
    except Exception as e:
        logger.warning(f"Qdrant delete for {doc_id} failed (may not be running): {e}")

    # Remove from doc_memory and rebuild
    mem_path = INDEX_DIR / "doc_memory.json"
    if mem_path.exists():
        data = json.loads(mem_path.read_text(encoding="utf-8"))
        data["docs"] = [d for d in data.get("docs", []) if d["doc_id"] != doc_id]
        mem_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Remove from SQLite
    storage.delete_document(doc_id)

    # Reload retriever indexes if running in-memory
    retriever_backend = CONFIG["retriever"].get("backend", "in_memory")
    if retriever_backend == "in_memory":
        try:
            from tools.colpali_tool import _state
            if _state.get("indexes") is not None and doc_id in _state["indexes"]:
                del _state["indexes"][doc_id]
                logger.info(f"Removed {doc_id} from in-memory retriever index")
        except Exception:
            pass

    return {"deleted": doc_id}


# ---------------------------------------------------------------------------
# P14: Upload endpoint
# ---------------------------------------------------------------------------

# In-memory upload progress tracking: upload_id → {doc_id, status, message, pct}
_upload_progress: dict[str, dict] = {}


@app.post("/api/v1/upload", response_model=UploadResponse)
async def upload_file_handler(req: Request):
    """Upload a PDF/image and start the ingestion pipeline asynchronously."""
    import tempfile
    import threading
    from fastapi.responses import JSONResponse

    try:
        form = await req.form()
        file = form.get("file")
        if not file:
            return JSONResponse(status_code=400, content={"detail": "No file provided"})

        filename = getattr(file, "filename", "upload.pdf")
        contents = await file.read()
        if not contents:
            return JSONResponse(status_code=400, content={"detail": "Empty file"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"detail": f"Invalid form: {e}"})

    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp"}:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Unsupported file type: {suffix}. Use PDF or image."},
        )

    # Derive doc_id and write to temp location
    from ingestion.upload import derive_upload_doc_id, run_upload_pipeline
    doc_id = derive_upload_doc_id(filename)
    upload_id = uuid.uuid4().hex[:12]

    _upload_progress[upload_id] = {
        "doc_id": doc_id,
        "status": "queued",
        "message": "准备中...",
        "pct": 0.0,
    }

    # Write file to temp path for pipeline
    temp_dir = Path(tempfile.gettempdir()) / "findoc_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{upload_id}_{filename}"
    temp_path.write_bytes(contents)

    def _progress_cb(stage: str, msg: str, pct: float):
        _upload_progress[upload_id] = {
            "doc_id": doc_id,
            "status": stage,
            "message": msg,
            "pct": pct,
        }

    def _run():
        try:
            run_upload_pipeline(str(temp_path), doc_id, progress_callback=_progress_cb)
        except Exception as e:
            logger.exception(f"Upload pipeline failed for {doc_id}")
            _upload_progress[upload_id] = {
                "doc_id": doc_id,
                "status": "failed",
                "message": f"{type(e).__name__}: {e}",
                "pct": 0.0,
            }
        finally:
            # Cleanup temp file
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()

    return UploadResponse(upload_id=upload_id, doc_id=doc_id, status="queued")


@app.get("/api/v1/upload/{upload_id}/status")
async def upload_status(upload_id: str):
    """SSE stream for upload progress."""
    import asyncio

    if upload_id not in _upload_progress:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "Upload not found"})

    async def event_stream():
        while True:
            info = _upload_progress.get(upload_id, {})
            status = info.get("status", "queued")

            payload = json.dumps({
                "doc_id": info.get("doc_id", ""),
                "status": status,
                "message": info.get("message", ""),
                "pct": info.get("pct", 0.0),
            }, ensure_ascii=False)
            yield f"data: {payload}\n\n"

            if status in ("done", "failed"):
                # Keep record for 1 hour then purge
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
