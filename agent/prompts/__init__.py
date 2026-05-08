"""Prompt loader with variant + few-shot support / Prompt 加载器——支持变体+few-shot。

Structure:
    prompts/
      system/persona.txt
      planner/{base,single_fact,cross_doc_compare,multi_step_calc,trend_analysis}.txt
      planner/few_shot/examples.jsonl
      verifier/{base,numeric,strict,factual}.txt
      synthesizer/{base,numeric}.txt

Usage:
    load_prompt("planner")                     -> base variant
    load_prompt("planner", variant="single_fact") -> single_fact variant
    load_prompt("planner", variant="single_fact", with_few_shot=True)
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load_prompt(node: str, variant: str = "base", with_few_shot: bool = False) -> str:
    """Load a prompt template for a given node and variant / 加载指定节点和变体的提示模板。

    Args / 参数:
        node: "planner" | "verifier" | "synthesizer" | "system"
        variant: "base" | "single_fact" | "cross_doc_compare" | "multi_step_calc"
                 | "trend_analysis" | "numeric" | "strict" | "factual"
        with_few_shot: prepend few-shot examples (only for planner variants) / 预置few-shot示例（仅用于planner变体）。
    """
    if node == "system":
        path = _HERE / "system" / "persona.txt"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    # Try variant-specific file first, fall back to base
    variant_path = _HERE / node / f"{variant}.txt"
    if variant_path.exists():
        text = variant_path.read_text(encoding="utf-8")
    else:
        base_path = _HERE / node / "base.txt"
        if base_path.exists():
            text = base_path.read_text(encoding="utf-8")
        else:
            # Legacy flat file fallback
            legacy = _HERE / f"{node}.txt"
            if legacy.exists():
                text = legacy.read_text(encoding="utf-8")
            else:
                raise FileNotFoundError(f"No prompt found for node={node}, variant={variant}")

    if with_few_shot and node == "planner":
        shots = _load_few_shots(variant)
        if shots:
            text = shots + "\n\n" + text

    return text


def _load_few_shots(variant: str) -> str:
    """Load few-shot examples matching a query_class variant from examples.jsonl / 从examples.jsonl加载匹配query_class变体的few-shot示例。

    Returns a formatted string of 2-3 examples, or empty string if not available / 返回2-3个格式化示例字符串，无可用示例时返回空字符串。
    """
    jsonl_path = _HERE / "planner" / "few_shot" / "examples.jsonl"
    if not jsonl_path.exists():
        return ""

    try:
        examples = []
        for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            ex = json.loads(line)
            if ex.get("query_class") == variant:
                examples.append(ex)
            elif variant == "base" and not ex.get("query_class"):
                examples.append(ex)
    except Exception:
        return ""

    if not examples:
        return ""

    lines = ["下面是几个示例，帮助你理解输出格式：\n"]
    for i, ex in enumerate(examples[:3], 1):
        lines.append(f"示例 {i}: 问：{ex['query']}")
        lines.append(f"答：{json.dumps({'plan': ex['plan'], 'query_class': ex['query_class']}, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


def list_variants(node: str) -> list[str]:
    """List available variant names for a node / 列出某个节点的可用变体名称。"""
    dir_path = _HERE / node
    if not dir_path.is_dir():
        return ["base"]
    variants = []
    for f in dir_path.glob("*.txt"):
        variants.append(f.stem)
    return sorted(variants) if variants else ["base"]
