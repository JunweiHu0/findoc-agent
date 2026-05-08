"""Structured fact extraction / 结构化事实提取工具。

Extracts entity, period, metric, value, and unit from free-text Fact entries
using regex patterns specialized for Chinese financial documents.
A lightweight LLM call can optionally refine the extraction.
"""
from __future__ import annotations

import re

from loguru import logger

from agent.state import Fact

# Common Chinese company name patterns
_COMPANY_RE = re.compile(
    r"(贵州茅台|宁德时代|招商银行|恒瑞医药|万科[ＡA]?|"
    r"比亚迪|美的集团|格力电器|海尔智家|"
    r"工商银行|建设银行|农业银行|中国银行|交通银行|"
    r"中国平安|中国人寿|中国太保|新华保险|"
    r"中信证券|海通证券|华泰证券|"
    r"[一-鿿]{2,6}(?:股份|集团|控股|科技|医药|银行|保险|证券|汽车|电器|地产))"
)

# Year / period patterns
_PERIOD_RE = re.compile(
    r"(20\d{2})(?:年(?:度|报)?)?(?:[Qq]([1-4])|([Hh]([12])))?"
)
# Simpler: just extract years
_YEAR_RE = re.compile(r"(20\d{2})")

# Metric keywords
_METRIC_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("营业收入", re.compile(r"营业(?:总)?收入")),
    ("营业成本", re.compile(r"营业(?:总)?成本")),
    ("毛利率", re.compile(r"毛利[率％%]?")),
    ("净利率", re.compile(r"净利[率％%]?")),
    ("净利润", re.compile(r"净利润")),
    ("研发费用", re.compile(r"研发(?:费用|投入|支出)")),
    ("研发占比", re.compile(r"研发(?:投入|费用|支出).*(?:占比|比例|占.*比)")),
    ("员工人数", re.compile(r"员[工工].*(?:人数|总数|数量)")),
    ("总资产", re.compile(r"(?:总)?资产(?:总额|总计)?")),
    ("净资产", re.compile(r"净资产")),
    ("ROE", re.compile(r"ROE|净资产收益率|净资产回报率")),
    ("ROA", re.compile(r"ROA|总资产收益率|总资产回报率")),
    ("每股收益", re.compile(r"每股收益|EPS")),
    ("资产负债率", re.compile(r"资产负债[率％%]")),
    ("经营活动现金流", re.compile(r"经营活动.*现金流")),
    ("货币资金", re.compile(r"货币资金")),
    ("存货", re.compile(r"存货(?:余额|金额)?")),
    ("应收账款", re.compile(r"应收账款(?:余额|金额)?")),
    ("分红", re.compile(r"(?:现金)?分红|股利|派息")),
]

# Number + unit patterns
_VALUE_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(亿元|万元|元|%|％|倍|亿|万|千|个|人)"
)

# Table row pattern
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|.*\|\s*$")


def extract_structured_facts(facts: list[Fact]) -> list[Fact]:
    """Enrich each Fact with structured entity/period/metric/value/unit fields / 为每个事实补充结构化字段（实体/期间/指标/数值/单位）。

    Returns the same list with modified Fact objects (in-place mutation for
    efficiency — the caller owns the list).
    """
    for fact in facts:
        try:
            _extract_one(fact)
        except Exception as e:
            logger.debug(f"fact extraction failed for {fact.source_doc} p.{fact.source_page}: {e}")
    return facts


def _extract_one(fact: Fact) -> None:
    """Extract structured fields from a single Fact / 从单个事实中提取结构化字段。"""
    text = fact.text

    # Entity: use source_doc as a hint, then regex
    doc_id = fact.source_doc
    entity = _extract_entity(text, doc_id)
    if entity:
        fact.entity = entity

    # Period: extract year / quarter
    period = _extract_period(text)
    if period:
        fact.period = period

    # Metric: match against known keywords
    metric = _extract_metric(text)
    if metric:
        fact.metric = metric

    # Value + unit
    value, unit = _extract_value_unit(text)
    if value is not None:
        fact.value = value
    if unit:
        fact.unit = unit

    # Classify
    if value is not None:
        fact.raw_kind = "numeric"
    elif _TABLE_ROW_RE.search(text):
        fact.raw_kind = "table_row"
    elif entity and (metric or period):
        fact.raw_kind = "string"
    else:
        fact.raw_kind = "unstructured"


def _extract_entity(text: str, doc_id: str) -> str | None:
    """Extract company/entity name from fact text."""
    # First priority: explicit company name in text
    m = _COMPANY_RE.search(text)
    if m:
        return m.group(1)

    # Fallback: derive from doc_id
    doc_map = {
        "moutai": "贵州茅台",
        "catl": "宁德时代",
        "cmb": "招商银行",
        "hengrui": "恒瑞医药",
        "vanke": "万科",
    }
    for prefix, name in doc_map.items():
        if doc_id.startswith(prefix):
            return name

    # Check if doc_id contains a readable name
    # (user-uploaded docs: e.g. balance_sheet_a3f9c1)
    stem = doc_id.rsplit("_", 1)[0] if "_" in doc_id else doc_id
    if len(stem) >= 2 and not stem.isalnum():
        return stem

    return None


def _extract_period(text: str) -> str | None:
    """Extract period (year, quarter, half) from fact text."""
    m = _PERIOD_RE.search(text)
    if m:
        year = m.group(1)
        if m.group(3):  # H1/H2
            half = m.group(4)
            return f"{year}H{half}"
        if m.group(2):  # Q1-Q4
            quarter = m.group(2)
            return f"{year}Q{quarter}"
        return year

    # Simpler year-only
    m = _YEAR_RE.search(text)
    if m:
        return m.group(1)

    return None


def _extract_metric(text: str) -> str | None:
    """Match fact text against known financial metric patterns."""
    for metric_name, pattern in _METRIC_PATTERNS:
        if pattern.search(text):
            return metric_name
    return None


def _extract_value_unit(text: str) -> tuple[float | None, str | None]:
    """Extract numeric value and Chinese unit from fact text.

    Returns (value, unit) or (None, None).
    When multiple numbers are present, picks the largest — financial facts
    tend to highlight the primary figure.
    """
    matches = _VALUE_UNIT_RE.findall(text)
    if not matches:
        # Try bare numbers without units
        bare = re.findall(r"(\d+(?:\.\d+)?)", text)
        if bare:
            try:
                return float(bare[0]), None
            except ValueError:
                pass
        return None, None

    # Pick the match with the largest numeric value
    best_val = float("-inf")
    best_unit = None
    for num_str, unit in matches:
        try:
            val = float(num_str)
            if val > best_val:
                best_val = val
                best_unit = unit
        except ValueError:
            continue

    if best_val == float("-inf"):
        return None, None
    return best_val, best_unit


def build_fact_index(facts: list[Fact]) -> dict:
    """Build a lookup index from extracted_facts keyed by (entity, period, metric) / 以 (实体, 期间, 指标) 为键构建事实查找索引。

    Used by verifier to mechanically check "do we already have (茅台, 2023, 毛利率)?"
    """
    index: dict = {}
    for f in facts:
        if not f.entity and not f.period and not f.metric:
            continue
        key = (f.entity or "", f.period or "", f.metric or "")
        if key not in index or (f.value is not None and index[key].value is None):
            index[key] = f
    return index
