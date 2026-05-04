"""AgentState — the shared TypedDict that flows through all LangGraph nodes.

Fields annotated with `Annotated[T, add]` use LangGraph's reducer pattern:
each node returns a delta, and the runtime appends to the list rather than
overwriting. This is how retrieved_pages, extracted_facts, tried_queries, etc.
accumulate across reflexion loops without explicit merging.
"""

from operator import add
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel


class SubTask(BaseModel):
    """A single step in the execution plan produced by the planner."""
    sub_query: str
    target_doc: Optional[str] = None
    expected_output_schema: str = "text"
    tool_calls: list[dict] = []  # P21: [{tool, args}] — takes priority over expected_output_schema when non-empty


class PageHit(BaseModel):
    """A page retrieved by ColQwen multi-vector MaxSim search."""
    doc_id: str
    page_num: int
    score: float = 0.0
    image_path: Optional[str] = None


class Fact(BaseModel):
    """A structured fact extracted from a page by VLM + fact_extractor (P24)."""
    text: str
    source_doc: str
    source_page: int
    sub_task_idx: Optional[int] = None
    # P24: structured extraction fields
    entity: Optional[str] = None       # company / entity name
    period: Optional[str] = None       # "2023" / "2023Q1" / "2023H1"
    metric: Optional[str] = None       # "营业收入" / "毛利率"
    value: Optional[float] = None
    unit: Optional[str] = None         # "亿元" / "%"
    raw_kind: Literal["numeric", "string", "table_row", "unstructured"] = "unstructured"


class ComputedValue(BaseModel):
    """A numeric result produced by the calculator tool (AST safe-eval)."""
    expr: str
    value: float
    sub_task_idx: Optional[int] = None


class Citation(BaseModel):
    """A (doc_id, page_num) pair referenced in the final answer."""
    doc_id: str
    page_num: int


class AgentState(TypedDict, total=False):
    """All state shared across the 7-node LangGraph pipeline.

    Lists marked `Annotated[..., add]` are auto-accumulated — nodes return
    deltas and the reducer merges them.
    """
    query: str

    plan: list[SubTask]
    plan_cursor: int

    # Accumulated evidence
    retrieved_pages: Annotated[list[PageHit], add]
    extracted_facts: Annotated[list[Fact], add]
    computed_values: Annotated[list[ComputedValue], add]

    # Reflexion control (P19–P20)
    reflexion_iter: int
    is_sufficient: bool
    missing_info: str  # deprecated free-text; kept for backward compat
    missing_facts: Annotated[list[dict], add]  # P19: structured [{sub_task_idx, what, root_cause, ...}]

    # P19: reflexion memory — prevents re-trying the same query / page
    tried_queries: Annotated[list[str], add]
    tried_pages: Annotated[list[dict], add]  # [{doc_id, page_num}]

    # P20: resource budget for reflexion loop
    budget_retrievals: int
    budget_vlm_calls: int

    # P20/21: remediation hint carried into executor
    remediation_hint: Optional[dict]  # {strategy, widened_top_k, suggested_page_nums, ...}

    # P22: retrieval scout candidates
    scout_candidates: list[dict]  # [{doc_id, top_page_num, top_score}]

    # P23: grounding audit
    unverified_claims: Annotated[list[dict], add]
    grounding_score: float

    # P24: structured fact index computed from extracted_facts each cycle
    fact_index: dict  # {(entity, period, metric): Fact}

    # P25: facts carried over from previous conversation turns
    known_facts: list[dict]

    # Final output
    answer: str
    citations: list[Citation]

    # Multi-turn context
    chat_history: list[dict]

    # User-scoped document filter (P14)
    doc_filter: Optional[list[str]]
