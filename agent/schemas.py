"""Pydantic schemas used as structured output targets for LLM nodes."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PlanItemSchema(BaseModel):
    sub_query: str = Field(description="Retrieval query, or arithmetic expression if expected_output_schema is 'number'.")
    target_doc: Optional[str] = Field(default=None, description="Restrict retrieval to this doc_id, or null for any.")
    expected_output_schema: str = Field(default="text", description='"number" | "text" | "table"')


class PlannerOutput(BaseModel):
    plan: list[PlanItemSchema]


class VerifierOutput(BaseModel):
    is_sufficient: bool = Field(description="True iff collected evidence is enough to answer the question.")
    inconsistency: str = Field(default="", description="Non-empty if numbers / calibers conflict across pages.")
    missing_info: str = Field(default="", description="When not sufficient, the next sub_query the executor should run.")
