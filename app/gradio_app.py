"""Gradio demo. P1: minimal chat surface that invokes the stub graph."""
from __future__ import annotations

import gradio as gr

from agent.graph import compile_graph


_app = compile_graph()


def _format_citations(citations) -> str:
    if not citations:
        return ""
    refs = ", ".join(f"[{c.doc_id} p.{c.page_num}]" for c in citations)
    return f"\n\n**引用**：{refs}"


def query_fn(message: str, history) -> str:
    init = {
        "query": message,
        "plan_cursor": 0,
        "reflexion_iter": 0,
        "is_sufficient": False,
        "retrieved_pages": [],
        "extracted_facts": [],
    }
    out = _app.invoke(init)
    return (out.get("answer") or "(no answer)") + _format_citations(out.get("citations") or [])


def main() -> None:
    gr.ChatInterface(
        fn=query_fn,
        title="FinDoc Agent",
        description="多模态金融文档问答（开发中）",
    ).launch()


if __name__ == "__main__":
    main()
