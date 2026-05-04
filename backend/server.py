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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from agent.config import CONFIG, INDEX_DIR, PAGES_DIR, ROOT
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
    if node == "retrieval_scout":
        candidates = delta.get("scout_candidates") or []
        if not candidates:
            return "未找到候选文档"
        top = candidates[0]
        return f"Top-{len(candidates)}候选: {top.get('doc_id','?')}"
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
        confidence = delta.get("confidence", 0.5)
        if delta.get("is_sufficient"):
            return f"✅ 充分 (置信{confidence:.0%})"
        missing = len(delta.get("missing_facts") or [])
        return f"↻ 缺{missing}项 (置信{confidence:.0%})"
    if node == "remediation":
        return "🔧 差异化修复"
    if node == "synthesizer":
        ans = delta.get("answer") or ""
        cites = len(delta.get("citations") or [])
        return f"生成答案({len(ans)}字符 · {cites}引用)"
    if node == "grounding":
        score = delta.get("grounding_score", 1.0)
        unverified = len(delta.get("unverified_claims") or [])
        if unverified == 0:
            return "✅ 校验通过"
        return f"⚠ 校验: {unverified}项未匹配"
    return "完成"


def _format_delta(node: str, delta: dict[str, Any]) -> str:
    if node == "retrieval_scout":
        candidates = delta.get("scout_candidates") or []
        if not candidates:
            return "*(未找到候选文档 — planner 将使用 doc_metadata)*"
        lines = [f"**检索前探查** — Top-{len(candidates)} 候选文档\n"]
        for i, c in enumerate(candidates, 1):
            lines.append(f"{i}. `{c.get('doc_id','?')}` 最佳页 p.{c.get('top_page_num','?')}  score={c.get('top_score',0):.4f}")
        return "\n".join(lines)
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
def _initial_state(
    query: str,
    doc_filter: list[str] | None = None,
    chat_history: list[dict] | None = None,
    known_facts: list[dict] | None = None,
) -> dict[str, Any]:
    return {
        "query": query,
        "plan_cursor": 0,
        "reflexion_iter": 0,
        "is_sufficient": False,
        "retrieved_pages": [],
        "extracted_facts": [],
        "computed_values": [],
        "doc_filter": doc_filter,
        "chat_history": chat_history or [],
        "tried_queries": [],
        "tried_pages": [],
        "missing_facts": [],
        "budget_retrievals": 10,
        "budget_vlm_calls": 20,
        "scout_candidates": [],
        "unverified_claims": [],
        "grounding_score": 0.0,
        "fact_index": {},
        "known_facts": known_facts or [],
    }


# P15: how many recent (user, assistant) pairs to inject into planner context
_HISTORY_TURNS = 4


async def _maybe_auto_title(conv_id: str, query: str, answer: str) -> None:
    """P17: After the first user turn finishes, ask the LLM for a short title.

    Fire-and-forget: any failure is swallowed — the conversation just keeps
    its 26-char query stub. Only triggered when the conversation has exactly
    one (user, assistant) pair, i.e., this was the first turn.
    """
    try:
        msgs = storage.get_messages(conv_id)
        if len(msgs) != 2:  # only act on the first complete turn
            return

        from agent.llm import get_llm, has_llm_key
        if not has_llm_key():
            return

        import asyncio as _aio

        def _summarize() -> str:
            llm = get_llm("synthesizer")
            prompt = (
                "请用不超过 12 个汉字给下面这段对话起一个标题，"
                "只返回标题本身，不要引号和句号。\n\n"
                f"问：{query.strip()[:120]}\n答：{answer.strip()[:200]}"
            )
            return llm.invoke(prompt).content.strip()

        title = (await _aio.to_thread(_summarize)).replace("\n", " ").strip()
        # Strip quotes / trailing punctuation the model sometimes emits anyway.
        title = title.strip("\"'`「」『』 \t").rstrip("。.！!？?")
        if not title:
            return
        title = title[:24]  # hard ceiling — protects sidebar layout
        storage.update_conversation_title(conv_id, title)
        logger.info(f"auto-titled conversation {conv_id}: {title}")
    except Exception as e:
        logger.debug(f"auto-title skipped for {conv_id}: {e}")


def _load_chat_history(conv_id: str) -> list[dict]:
    """Pull the last _HISTORY_TURNS user+assistant pairs from SQLite.

    Returns a list of {role, content} dicts in chronological order
    (oldest-first), suitable for injection into AgentState.chat_history.
    """
    if not conv_id:
        return []
    try:
        msgs = storage.get_messages(conv_id)
    except Exception as e:
        logger.warning(f"failed to load chat history for {conv_id}: {e}")
        return []
    # Keep only the trailing 2*_HISTORY_TURNS messages (one pair = 2 messages).
    tail = msgs[-(2 * _HISTORY_TURNS):]
    return [{"role": m["role"], "content": m["content"]} for m in tail]


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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    token_queue: queue.Queue[str] = queue.Queue()  # P16: synthesizer streaming

    def _on_progress(msg: str) -> None:
        progress_queue.put(msg)

    def _on_token(tok: str) -> None:
        token_queue.put(tok)

    def _run_agent_sync(state: dict) -> None:
        """Execute agent graph synchronously in a background thread."""
        from agent.nodes.synthesizer import set_token_hook
        from tools.colpali_tool import set_progress_hook as set_colpali_hook
        from tools.vlm_tool import set_progress_hook as set_vlm_hook
        set_colpali_hook(_on_progress)
        set_vlm_hook(_on_progress)
        set_token_hook(_on_token)

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
        history = _load_chat_history(conv_id)
        # P25: load cross-turn facts
        known_facts = storage.load_conv_facts(conv_id) if conv_id else []
        state = _initial_state(req.query, req.doc_filter, history, known_facts)
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

            # Drain synthesizer tokens (P16) — emitted while synthesizer node is still running.
            had_token = False
            while True:
                try:
                    tok = token_queue.get_nowait()
                    payload = json.dumps({"type": "token", "token": tok}, ensure_ascii=False)
                    yield f"event: token\ndata: {payload}\n\n"
                    had_token = True
                except queue.Empty:
                    break

            # Drain node results
            try:
                node_name, data = node_queue.get_nowait()
            except queue.Empty:
                if not had_progress and not had_token:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.1 if had_token else 0.3)
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

        # Final drain of any tokens that arrived after the synthesizer completed
        # but before we read the __done__ marker.
        while True:
            try:
                tok = token_queue.get_nowait()
                payload = json.dumps({"type": "token", "token": tok}, ensure_ascii=False)
                yield f"event: token\ndata: {payload}\n\n"
            except queue.Empty:
                break

        accumulated: dict[str, Any] = data  # data from __done__
        answer = (accumulated.get("answer") or "").strip()
        citations = accumulated.get("citations") or []
        pages = accumulated.get("retrieved_pages") or []
        grounding_score = accumulated.get("grounding_score", 1.0)
        unverified_claims = accumulated.get("unverified_claims") or []

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

        # P25: persist structured facts for cross-turn reuse
        try:
            extracted = accumulated.get("extracted_facts") or []
            fact_dicts = [f.model_dump() if hasattr(f, "model_dump") else f for f in extracted]
            storage.save_conv_facts(conv_id, fact_dicts)
        except Exception as e:
            logger.warning(f"Failed to save conv_facts for {conv_id}: {e}")

        # P17: fire-and-forget auto-title once we have a (user, assistant) pair.
        asyncio.create_task(_maybe_auto_title(conv_id, req.query, answer))

        done = json.dumps({
            "type": "done",
            "conv_id": conv_id,
            "answer": answer,
            "citations": [_citation_to_out(c).model_dump() for c in citations],
            "retrieved_pages": [_page_hit_to_out(p).model_dump() for p in pages],
            "grounding_score": grounding_score,
            "unverified_claims": unverified_claims,
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

_PAGE_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_THUMB_FILENAME = "_thumb.jpg"
_THUMB_MAX = 480              # max edge in px — sidebar card is ~300px, leaves HiDPI headroom
_THUMB_QUALITY = 78           # JPEG quality
_THUMB_CACHE_HEADERS = {"Cache-Control": "public, max-age=86400"}


def _find_page_image(doc_id: str) -> Path | None:
    """Return the first page image for this doc, regardless of naming.

    Scans the doc's pages dir for any *.png / *.jpg / *.jpeg / *.webp file
    (excluding our own cached `_thumb.jpg`). Returns None when the dir is
    missing or empty — the frontend then renders a plain text fallback.
    """
    pages_dir = PAGES_DIR / doc_id
    if not pages_dir.exists():
        return None
    candidates = [
        p for p in pages_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in _PAGE_IMG_EXTS
        and p.name != _THUMB_FILENAME
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)  # p001 < p002 < ...
    return candidates[0]


def _build_thumb_data_url(doc_id: str) -> str | None:
    """Generate (or read cached) thumbnail and return it as a base64 data URL.

    Inlining the bytes in the /documents response eliminates the N follow-up
    HTTP requests the panel previously made — one request now returns
    everything the sidebar needs to render.
    """
    import base64

    source = _find_page_image(doc_id)
    if source is None:
        return None

    thumb = source.parent / _THUMB_FILENAME
    try:
        if not thumb.exists() or thumb.stat().st_mtime < source.stat().st_mtime:
            from PIL import Image
            with Image.open(source) as im:
                im = im.convert("RGB")
                im.thumbnail((_THUMB_MAX, _THUMB_MAX), Image.LANCZOS)
                im.save(thumb, "JPEG", quality=_THUMB_QUALITY, optimize=True, progressive=True)
        b64 = base64.b64encode(thumb.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        logger.warning(f"thumbnail generation failed for {doc_id}: {e}")
        return None


@app.get("/api/v1/documents", response_model=list[DocumentOut])
async def list_documents_all():
    """List user-uploaded documents with inlined thumbnails.

    Default offline indexes remain queryable but are not part of the user
    document management surface. Thumbnails are generated in parallel on a
    thread pool (PIL releases the GIL during decode/resize) and inlined as
    base64 so the panel needs only this one request to render.
    """
    import asyncio

    docs = storage.list_documents()
    uploads_dir = ROOT / "data" / "uploads"
    user_docs = [d for d in docs if (uploads_dir / d["doc_id"]).exists()]

    if user_docs:
        loop = asyncio.get_event_loop()
        thumbs = await asyncio.gather(*[
            loop.run_in_executor(None, _build_thumb_data_url, d["doc_id"])
            for d in user_docs
        ])
        for d, t in zip(user_docs, thumbs):
            d["thumbnail"] = t

    return [DocumentOut(**d) for d in user_docs]


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
# P18: Reindex + cover preview
# ---------------------------------------------------------------------------

@app.post("/api/v1/documents/{doc_id}/reindex", response_model=UploadResponse)
async def reindex_doc(doc_id: str):
    """Re-run the upload pipeline for a previously uploaded document.

    Source file is read from data/uploads/<doc_id>/. Returns the same
    UploadResponse shape as /upload so the frontend can subscribe to
    /upload/{id}/status to track progress.
    """
    import threading
    from fastapi.responses import JSONResponse
    from ingestion.upload import UPLOAD_DIR, run_upload_pipeline

    doc_upload_dir = UPLOAD_DIR / doc_id
    if not doc_upload_dir.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": f"No source file for {doc_id} — original upload not retained."},
        )

    # Pick the first regular file in the upload dir as the source.
    sources = [p for p in doc_upload_dir.iterdir() if p.is_file()]
    if not sources:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Upload dir empty for {doc_id}."},
        )
    source = sources[0]
    upload_id = uuid.uuid4().hex[:12]

    _purge_stale_uploads()
    _upload_progress[upload_id] = {
        "doc_id": doc_id,
        "status": "queued",
        "message": "重新索引排队中...",
        "pct": 0.0,
    }

    def _progress_cb(stage: str, msg: str, pct: float):
        import time as _t
        entry = {"doc_id": doc_id, "status": stage, "message": msg, "pct": pct}
        if stage in ("done", "failed"):
            entry["finished_at"] = _t.time()
        _upload_progress[upload_id] = entry

    def _run():
        import time as _t
        try:
            run_upload_pipeline(str(source), doc_id, progress_callback=_progress_cb)
        except Exception as e:
            logger.exception(f"Reindex pipeline failed for {doc_id}")
            _upload_progress[upload_id] = {
                "doc_id": doc_id, "status": "failed",
                "message": f"{type(e).__name__}: {e}", "pct": 0.0,
                "finished_at": _t.time(),
            }

    threading.Thread(target=_run, daemon=True).start()
    return UploadResponse(upload_id=upload_id, doc_id=doc_id, status="queued")


@app.get("/api/v1/documents/{doc_id}/cover")
async def doc_cover(doc_id: str):
    """Standalone cover endpoint — kept for backward compatibility.

    The sidebar inlines thumbnails through /api/v1/documents directly and
    no longer hits this route. Existing callers still get a JPEG thumbnail.
    """
    from fastapi.responses import FileResponse, JSONResponse

    source = _find_page_image(doc_id)
    if source is None:
        return JSONResponse(status_code=404, content={"detail": "No image available"})

    thumb = source.parent / _THUMB_FILENAME
    try:
        if not thumb.exists() or thumb.stat().st_mtime < source.stat().st_mtime:
            from PIL import Image
            with Image.open(source) as im:
                im = im.convert("RGB")
                im.thumbnail((_THUMB_MAX, _THUMB_MAX), Image.LANCZOS)
                im.save(thumb, "JPEG", quality=_THUMB_QUALITY, optimize=True, progressive=True)
    except Exception as e:
        logger.warning(f"thumbnail generation failed for {doc_id}: {e} — falling back to source")
        return FileResponse(str(source), headers=_THUMB_CACHE_HEADERS)

    return FileResponse(str(thumb), media_type="image/jpeg", headers=_THUMB_CACHE_HEADERS)


# ---------------------------------------------------------------------------
# P14: Upload endpoint
# ---------------------------------------------------------------------------

# In-memory upload progress tracking: upload_id → {doc_id, status, message, pct, finished_at}
# Terminal entries (done/failed) are purged after _UPLOAD_TTL_SEC.
_upload_progress: dict[str, dict] = {}
_UPLOAD_TTL_SEC = 3600  # 1 hour


def _purge_stale_uploads() -> None:
    """Drop terminal upload records older than TTL. Called opportunistically
    on each new upload — no background task needed for a single-user app."""
    import time
    now = time.time()
    stale = [
        uid for uid, info in _upload_progress.items()
        if info.get("finished_at") and now - info["finished_at"] > _UPLOAD_TTL_SEC
    ]
    for uid in stale:
        _upload_progress.pop(uid, None)


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

    _purge_stale_uploads()
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
        import time
        entry = {
            "doc_id": doc_id,
            "status": stage,
            "message": msg,
            "pct": pct,
        }
        if stage in ("done", "failed"):
            entry["finished_at"] = time.time()
        _upload_progress[upload_id] = entry

    def _run():
        import time
        try:
            run_upload_pipeline(str(temp_path), doc_id, progress_callback=_progress_cb)
        except Exception as e:
            logger.exception(f"Upload pipeline failed for {doc_id}")
            _upload_progress[upload_id] = {
                "doc_id": doc_id,
                "status": "failed",
                "message": f"{type(e).__name__}: {e}",
                "pct": 0.0,
                "finished_at": time.time(),
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
