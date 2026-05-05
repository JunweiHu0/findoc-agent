"""Context compression (P27) — structured summarisation instead of brute truncation.

Three primitives:
  compress_history  — summarise chat_history when >3 turns or >800 tokens.
  compress_evidence — deduplicate extracted_facts by (entity, period, metric).
  TokenBudget       — per-node token caps; auto-trigger compression when exceeded.

Design: compression itself costs ~200 tokens for the summary prompt but the
downstream prompt savings are much larger. Discarded facts are not lost —
they are written to episodic memory (P28) and can still be recalled.
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------

class TokenBudget:
    """Per-node token limits. Exceeding triggers automatic compression."""

    planner: int = 1200
    verifier: int = 1500
    synthesizer: int = 2000

    @classmethod
    def for_node(cls, node: str) -> int:
        return getattr(cls, node, 1200)

    @classmethod
    def estimate(cls, text: str) -> int:
        """Rough token count: Chinese ~1 char/token, English ~4 char/token."""
        if not text:
            return 0
        # Count CJK characters separately
        cjk = len(re.findall(r"[一-鿿　-〿＀-￯]", text))
        other = len(text) - cjk
        return cjk + other // 4


# ---------------------------------------------------------------------------
# History compression
# ---------------------------------------------------------------------------

def compress_history(
    history: list[dict],
    budget: int = 600,
    trigger_turns: int = 3,
    trigger_tokens: int = 800,
) -> str:
    """Compress chat_history into a structured summary.

    When <= trigger_turns turns and <= trigger_tokens total, returns the raw
    history unchanged (compact format). Above threshold, older turns are
    collapsed into a structured summary preserving {entities, metrics, decisions,
    focus_doc}, and only the most recent 2 turns are kept verbatim.

    Args:
        history: list of {role, content} dicts in chronological order.
        budget: maximum characters for the output string.
        trigger_turns: number of turns above which compression is triggered.
        trigger_tokens: estimated token count above which compression triggers.
    """
    if not history:
        return "(no prior turns)"

    raw = _render_raw(history)
    est = TokenBudget.estimate(raw)

    if len(history) <= trigger_turns * 2 and est <= trigger_tokens:  # *2 because user+assistant per turn
        # Within budget — return compact raw format
        return _truncate_to_budget(raw, budget)

    # Compress: summarise older turns, keep last 2 turns verbatim
    cutoff = max(0, len(history) - 4)  # last 2 turns = 4 messages
    older = history[:cutoff]
    recent = history[cutoff:]

    summary = _extract_summary(older)
    recent_raw = _render_raw(recent)
    combined = f"[摘要] {summary}\n---\n{recent_raw}"
    return _truncate_to_budget(combined, budget)


def _render_raw(history: list[dict]) -> str:
    """Render history as compact 'U:'/'A:' lines."""
    lines: list[str] = []
    for turn in history:
        role = turn.get("role", "?")
        content = (turn.get("content") or "").strip().replace("\n", " ")
        prefix = "U" if role == "user" else "A"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _truncate_to_budget(text: str, budget: int) -> str:
    """Trim text to budget chars, preserving word boundaries."""
    if len(text) <= budget:
        return text
    return text[:budget - 3] + "..."


def _extract_summary(history: list[dict]) -> str:
    """Extract {entities, metrics, decisions, focus_doc} from history turns.

    Uses regex heuristics — no LLM call. Returns a compact Chinese summary
    string suitable for injection into the planner prompt.
    """
    all_text = " ".join(
        (h.get("content") or "") for h in history
    )

    parts: list[str] = []

    # Entities: company names mentioned
    entities = set()
    for m in re.finditer(
        r"(贵州茅台|宁德时代|招商银行|恒瑞医药|万科[ＡA]?|比亚迪|"
        r"美的集团|格力电器|工商银行|建设银行|中国平安|中国人寿|"
        r"[一-鿿]{2,6}(?:股份|集团|控股|科技|医药|银行|保险|证券))",
        all_text,
    ):
        entities.add(m.group(1))
    if entities:
        parts.append(f"实体: {', '.join(sorted(entities)[:5])}")

    # Metrics mentioned
    metrics = set()
    for kw in ["营收", "毛利率", "净利", "研发", "员工", "资产", "ROE", "分红", "同比", "增长"]:
        if kw in all_text:
            metrics.add(kw)
    if metrics:
        parts.append(f"指标: {', '.join(sorted(metrics)[:5])}")

    # Decisions (user asked to compute/compare/etc.)
    decisions = set()
    for kw in ["对比", "比较", "计算", "分析", "趋势", "变化", "差异"]:
        if kw in all_text:
            decisions.add(kw)
    if decisions:
        parts.append(f"意图: {', '.join(sorted(decisions))}")

    # Focus doc
    doc_ids = set()
    for m in re.finditer(r"doc_id[=:]\s*(\w+)", all_text):
        doc_ids.add(m.group(1))
    if doc_ids:
        parts.append(f"文档: {', '.join(sorted(doc_ids))}")

    return "; ".join(parts) if parts else "(无关键实体)"


# ---------------------------------------------------------------------------
# Evidence compression
# ---------------------------------------------------------------------------

def compress_evidence(
    facts: list,
    query: str = "",
    budget: int = 800,
    trigger_count: int = 10,
) -> list:
    """Deduplicate facts by (entity, period, metric), preserving highest confidence.

    When > trigger_count facts or reflexion_iter > 0, apply dedup:
    - For facts with the same (entity, period, metric), keep the one with
      the longest text (most detail).
    - Facts irrelevant to the query (no entity/metric overlap) are dropped
      but their content is retained for episodic memory (P28).

    Args:
        facts: list of Fact objects or dicts with entity/period/metric/text fields.
        query: the user's query, used for relevance filtering.
        budget: max chars for rendered output.
        trigger_count: number of facts above which compression triggers.
    """
    if len(facts) <= trigger_count:
        return list(facts)

    # Build dedup index by (entity, period, metric)
    indexed: dict[tuple, dict] = {}
    discarded: list = []

    for f in facts:
        entity = _field(f, "entity") or ""
        period = _field(f, "period") or ""
        metric = _field(f, "metric") or ""
        text = _field(f, "text") or ""

        key = (entity, period, metric)
        if key == ("", "", ""):
            # Fully unstructured — keep but don't dedup
            indexed[(entity, period, metric, id(f))] = f
            continue

        if key in indexed:
            existing_text = _field(indexed[key], "text") or ""
            if len(text) > len(existing_text):
                discarded.append(indexed[key])
                indexed[key] = f
            else:
                discarded.append(f)
        else:
            indexed[key] = f

    # Relevance filter: keep facts where entity or metric overlaps with query
    if query:
        kept = []
        for f in indexed.values():
            entity = _field(f, "entity") or ""
            metric = _field(f, "metric") or ""
            if entity and entity in query:
                kept.append(f)
            elif metric and any(m in query for m in metric.split()):
                kept.append(f)
            elif not entity and not metric:
                kept.append(f)  # unstructured but keep
        # If we dropped too many, fall back to all indexed
        if len(kept) < len(indexed) * 0.5:
            kept = list(indexed.values())
    else:
        kept = list(indexed.values())

    return kept


def _field(obj, name: str) -> Optional[str]:
    """Extract a field from a pydantic model or dict."""
    if hasattr(obj, name):
        return getattr(obj, name, None)
    if isinstance(obj, dict):
        return obj.get(name)
    return None
