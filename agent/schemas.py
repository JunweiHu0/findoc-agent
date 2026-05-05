"""Pydantic schemas used as structured output targets for LLM nodes.

PlannerOutput and VerifierOutput are passed to ChatOpenAI.with_structured_output()
to force the LLM to produce valid JSON matching these shapes (json_mode).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation planned by the planner node (P21)."""
    tool: str = Field(description="Tool name as listed in the Available Tools registry.")
    args: dict = Field(default_factory=dict, description="Keyword arguments matching the tool's params_schema.")


class PlanItemSchema(BaseModel):
    """One step in the planner's execution plan (P29: DAG-aware)."""
    sub_query: str = Field(description="Retrieval query, or arithmetic expression if expected_output_schema is 'number'.")
    target_doc: Optional[str] = Field(default=None, description="Restrict retrieval to this doc_id, or null for any.")
    expected_output_schema: str = Field(default="text", description='"number" | "text" | "table"')
    tool_calls: list[ToolCall] = Field(default_factory=list, description="P21: explicit tool dispatch list.")
    # P29: DAG fields
    task_id: str = Field(default="", description="Unique id for DAG scheduling.")
    depends_on: list[str] = Field(default_factory=list, description="Task IDs this step must wait for.")
    priority: Literal[0, 1] = Field(default=0, description="0=core, 1=exploratory.")


class PlannerOutput(BaseModel):
    """Structured output from the planner LLM call."""
    plan: list[PlanItemSchema]
    query_class: Optional[str] = Field(default=None, description="P22: one of single_fact | cross_doc_compare | multi_step_calc | trend_analysis")


# ---------------------------------------------------------------------------
# P19: Structured verifier output — root-cause-aware missing facts
# ---------------------------------------------------------------------------

class MissingFact(BaseModel):
    """A gap diagnosed by the verifier, with root cause and suggested remedy."""
    sub_task_idx: int = Field(description="Index (0-based) of the sub-task that left a gap.")
    what: str = Field(description="Human-readable description of what is missing.")
    root_cause: Literal[
        "retrieval_miss",       # ColQwen did not recall the right page
        "reading_miss",         # correct page recalled but VLM missed / misread
        "ambiguous_query",      # the sub_query itself was ambiguous
        "inconsistency",        # cross-page numbers conflict — need caliber / authority resolution
    ] = Field(description="Root cause category — drives downstream remediation strategy (P20).")
    suggested_query: Optional[str] = Field(default=None, description="Rewritten query for re-retrieval (retrieval_miss / ambiguous_query).")
    suggested_target_doc: Optional[str] = Field(default=None, description="Constrain re-retrieval to this doc_id.")
    suggested_page_nums: Optional[list[int]] = Field(default=None, description="Re-read only these page numbers (reading_miss).")


class TodoItemSchema(BaseModel):
    """Schema for runtime todo tracking (P26.5)."""
    id: str = Field(default="")
    sub_task_idx: int = Field(default=0)
    title: str = Field(default="")
    status: Literal["pending", "running", "done", "failed", "skipped"] = Field(default="pending")
    attempt: int = Field(default=0)
    error: Optional[str] = Field(default=None)
    parent_id: Optional[str] = Field(default=None)


class VerifierOutput(BaseModel):
    """Structured output from the verifier LLM call (P19)."""
    is_sufficient: bool = Field(description="True iff collected evidence is enough to answer the question.")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="How confident the verifier is in the sufficiency judgment (P19).")
    inconsistency: str = Field(default="", description="Non-empty if numbers / calibers conflict across pages.")
    missing_facts: list[MissingFact] = Field(default_factory=list, description="When not sufficient, structured list of what is missing and why (P19).")
    missing_info: str = Field(default="", description="Deprecated free-text fallback; prefer missing_facts.")
