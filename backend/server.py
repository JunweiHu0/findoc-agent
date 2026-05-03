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
from pathlib import Path
from typing import Any

# Bootstrap project root for agent.* imports
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from loguru import logger

from agent.config import CONFIG, INDEX_DIR
from agent.graph import compile_graph
from agent.state import Citation, PageHit
from backend.schemas import CitationOut, DocInfo, HealthResponse, PageHitOut, QueryRequest

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

        done = json.dumps({
            "type": "done",
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
