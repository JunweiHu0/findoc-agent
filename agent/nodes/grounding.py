"""Grounding node — post-hoc citation and numeric audit with feedback writeback / 后置校验节点——引用+数值审计，含验证结果回写。

Runs after synthesizer and before END. Validates:
1. Every `[doc_id p.N]` citation in the answer corresponds to an actual
   (doc_id, page_num) pair present in extracted_facts.
2. Every number in the answer fuzzy-matches at least one fact value.

Strips unverified citations and inserts a confidence banner when checks fail.
All checks are pure regex + set lookups — no LLM calls.
"""

from __future__ import annotations

import re
from typing import Optional

from loguru import logger

from ..state import AgentState

# Matches [doc_id p.N] citations — doc_id is alphanumeric + underscore
_CITATION_RE = re.compile(r"\[(\w+)\s+p\.(\d+)\]")

# Matches numeric values with Chinese units
_NUMERIC_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:亿元|万元|元|%|倍|亿|万|千|百|十|％)"
)

# Tolerance for fuzzy numeric matching (±0.1%)
_TOLERANCE = 0.001


def grounding_node(state: AgentState) -> dict:
    """Run citation + numeric audit on the final answer. Strip unverified claims and insert confidence banner. / 对最终答案运行引用+数值审计，剥离未验证声明并插入置信度标识。"""
    answer = state.get("answer") or ""
    if not answer:
        return {"grounding_score": 1.0, "unverified_claims": []}

    facts = state.get("extracted_facts") or []
    unverified: list[dict] = []

    # Build fact lookup sets
    known_pages: set[tuple[str, int]] = set()
    known_numbers: list[tuple[float, str]] = []  # (value, text)
    for f in facts:
        known_pages.add((f.source_doc, f.source_page))
        # Collect numbers from fact text
        for m in _NUMERIC_RE.finditer(f.text):
            try:
                val = float(m.group(1))
                known_numbers.append((val, f.text))
            except ValueError:
                pass
        # Also use structured fact value if available
        if f.value is not None:
            known_numbers.append((f.value, f.text))

    # 1. Citation check
    citations = _CITATION_RE.findall(answer)
    for doc_id, page_str in citations:
        page_num = int(page_str)
        if (doc_id, page_num) not in known_pages:
            unverified.append({
                "text": f"[{doc_id} p.{page_num}]",
                "reason": "citation_not_in_evidence",
            })

    # 2. Numeric check
    for m in _NUMERIC_RE.finditer(answer):
        full_match = m.group(0)
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        matched = _fuzzy_match(val, known_numbers)
        if not matched:
            unverified.append({
                "text": full_match,
                "reason": "number_not_in_evidence",
            })

    # Compute grounding score
    total_claims = len(citations) + len(list(_NUMERIC_RE.finditer(answer)))
    if total_claims == 0 or (not citations and not list(_NUMERIC_RE.finditer(answer))):
        grounding_score = 1.0
    else:
        grounding_score = 1.0 - (len(unverified) / max(total_claims, 1))
    grounding_score = max(0.0, min(1.0, grounding_score))

    # 3. Modify answer: strip unverified citations, add banner
    clean_answer = _strip_unverified_citations(answer, unverified)
    clean_answer = _add_grounding_banner(clean_answer, grounding_score, unverified)

    # feedback writeback — mark matched facts as verified / 反馈写回——标记匹配的事实为已验证
    _writeback_verification(state, unverified)

    logger.info(
        f"grounding: score={grounding_score:.2f}, "
        f"citations={len(citations)}, numbers={len(list(_NUMERIC_RE.finditer(answer)))}, "
        f"unverified={len(unverified)}"
    )

    return {
        "answer": clean_answer,
        "grounding_score": grounding_score,
        "unverified_claims": unverified,
    }


def _writeback_verification(state: AgentState, unverified: list[dict]) -> None:
    """Write grounding verification results back to conv_facts for memory promotion / 将校验结果写回 conv_facts 用于记忆晋升。"""
    try:
        from backend.storage import mark_conv_fact_verified, load_conv_facts, upsert_global_fact
        from agent.config import CONFIG

        facts = state.get("extracted_facts") or []
        if not facts:
            return

        # Collect unverified citation strings for quick lookup
        bad_cites = {
            u["text"] for u in unverified
            if u.get("reason") == "citation_not_in_evidence"
        }

        for f in facts:
            entity = (f.entity or "").strip()
            period = (f.period or "").strip()
            metric = (f.metric or "").strip()
            if not entity and not period and not metric:
                continue

            cite_str = f"[{f.source_doc} p.{f.source_page}]"
            verified = cite_str not in bad_cites

            # Load the fact from DB to get its id
            stored = load_conv_facts(state.get("conv_id", ""))
            for sf in stored:
                if (
                    sf.get("entity") == entity
                    and sf.get("period") == period
                    and sf.get("metric") == metric
                    and abs((sf.get("value") or 0) - (f.value or 0)) < 0.001
                ):
                    mark_conv_fact_verified(sf.get("id", 0), verified=verified)
                    # Check promotion threshold
                    min_hits = CONFIG.get("agent", {}).get("semantic_memory_min_hits", 3)
                    if verified and (sf.get("hit_count", 0) + 1) >= min_hits:
                        upsert_global_fact({
                            "entity": entity, "period": period, "metric": metric,
                            "value": f.value, "unit": f.unit or "",
                            "source_doc": f.source_doc, "source_page": f.source_page,
                            "text": f.text, "grounding_verified": 1,
                        })
                    break
    except Exception as e:
        logger.debug(f"grounding writeback skipped: {e}")


def _fuzzy_match(value: float, known: list[tuple[float, str]]) -> Optional[tuple[float, str]]:
    """Check if a numeric value matches any known fact within tolerance / 检查数值是否在容差内匹配已知事实。"""
    if value == 0:
        return (0.0, "") if any(abs(kv[0]) < 1e-9 for kv in known) else None
    for kv, text in known:
        if kv == 0:
            continue
        if abs(value - kv) / abs(kv) <= _TOLERANCE:
            return (kv, text)
    # Also try exact match
    for kv, text in known:
        if abs(value - kv) < 1e-6:
            return (kv, text)
    return None


def _strip_unverified_citations(answer: str, unverified: list[dict]) -> str:
    """Remove citations that failed verification from the answer / 从答案中移除未通过校验的引用。"""
    bad_cites = {
        u["text"] for u in unverified
        if u.get("reason") == "citation_not_in_evidence"
    }
    if not bad_cites:
        return answer
    for cite in bad_cites:
        answer = answer.replace(cite, "")
    # Clean up double spaces
    answer = re.sub(r"  +", " ", answer)
    return answer


def _add_grounding_banner(answer: str, score: float, unverified: list[dict]) -> str:
    """Prepend a confidence banner when grounding checks reveal issues / 校验发现问题时前置置信度标识。"""
    unverified_ratio = len(unverified) / max(len(_CITATION_RE.findall(answer)) + len(list(_NUMERIC_RE.finditer(answer))), 1)

    if score >= 0.95 and not unverified:
        return answer

    if score >= 0.7:
        banner = (
            "\n\n---\n"
            "⚠ **部分引用/数值未经证据校验**，请人工核对。\n"
        )
    else:
        banner = (
            "\n\n---\n"
            "🛑 **答案可信度低** — 多项引用或数值与检索证据不匹配，请人工核对。\n"
        )

    # List specific issues
    if unverified:
        banner += "\n未匹配项:\n"
        for u in unverified[:5]:  # cap at 5
            banner += f"- `{u['text']}` ({u['reason']})\n"
        if len(unverified) > 5:
            banner += f"- ... 等共 {len(unverified)} 项\n"

    return answer + banner
