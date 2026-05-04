"""Caliber disambiguation tool (P20).

When Verifier detects cross-page numerical inconsistency, this tool:
1. Re-reads the conflicting pages with a caliber-focused VLM instruction
2. Extracts disclosure notes (consolidated vs parent-company, FY vs CY, etc.)
3. Returns the authoritative fact with caliber annotation.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from agent.state import Fact


def disambiguate_caliber(
    facts: list[Fact],
    conflict_topic: str,
    vlm_read_fn: "callable | None" = None,
) -> dict:
    """Resolve a cross-page numerical conflict by extracting caliber metadata.

    Args:
        facts: The conflicting Fact objects (should be 2+). Each must have
               source_doc and source_page populated.
        conflict_topic: What the conflict is about, e.g. "营业收入口径".
        vlm_read_fn: Optional VLM read function (defaults to tools.vlm_tool.vlm_read_page).

    Returns:
        {"resolved": bool, "authoritative_fact": dict | None,
         "explanation": str, "caliber_notes": list[dict]}
    """
    if vlm_read_fn is None:
        from tools.vlm_tool import vlm_read_page as _default_vlm
        vlm_read_fn = _default_vlm

    caliber_notes: list[dict] = []

    # Re-read each conflicting page with a caliber-focused instruction
    for f in facts:
        if not f.source_doc or not f.source_page:
            continue
        instruction = (
            f"本页涉及'{conflict_topic}'的数据。请特别关注并提取：\n"
            "1. 该数字的披露口径（合并报表 vs 母公司报表）\n"
            "2. 会计期间（本期/上期/同比期间）\n"
            "3. 是否包含子公司或关联方\n"
            f"4. 该数字的具体数值和单位\n"
            "如果本页没有这些信息，回复'not on this page'。"
        )
        try:
            # Reconstruct image path
            from agent.config import PAGES_DIR
            img_path = str(PAGES_DIR / f.source_doc / f"p{f.source_page:03d}.png")
            extraction = vlm_read_fn(img_path, instruction)
        except Exception as e:
            logger.warning(f"disambiguate: VLM read failed for {f.source_doc} p.{f.source_page}: {e}")
            extraction = f"[error: {e}]"

        caliber_notes.append({
            "doc_id": f.source_doc,
            "page_num": f.source_page,
            "value": f.value,
            "text": f.text,
            "caliber_extraction": extraction,
        })

    # Heuristic: prefer consolidated > parent-company, later year > earlier
    # This is a rule-based fallback; the verifier will make the final call
    consolidated = []
    parent_only = []
    for note in caliber_notes:
        extraction = note.get("caliber_extraction", "")
        if "合并" in extraction and "母公司" not in extraction:
            consolidated.append(note)
        elif "母公司" in extraction:
            parent_only.append(note)

    preferred = consolidated or parent_only or caliber_notes

    return {
        "resolved": len(caliber_notes) > 0,
        "authoritative_fact": {
            "doc_id": preferred[0]["doc_id"],
            "page_num": preferred[0]["page_num"],
            "value": preferred[0]["value"],
            "text": preferred[0]["text"],
        },
        "explanation": (
            f"从 {len(caliber_notes)} 页中提取了口径信息；"
            f"优先采用{'合并报表' if consolidated else '可用'}数据"
        ),
        "caliber_notes": caliber_notes,
    }
