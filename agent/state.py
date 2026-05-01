from operator import add
from typing import Annotated, Optional, TypedDict

from pydantic import BaseModel


class SubTask(BaseModel):
    sub_query: str
    target_doc: Optional[str] = None
    expected_output_schema: str = "text"


class PageHit(BaseModel):
    doc_id: str
    page_num: int
    score: float = 0.0
    image_path: Optional[str] = None


class Fact(BaseModel):
    text: str
    source_doc: str
    source_page: int
    sub_task_idx: Optional[int] = None


class ComputedValue(BaseModel):
    expr: str
    value: float
    sub_task_idx: Optional[int] = None


class Citation(BaseModel):
    doc_id: str
    page_num: int


class AgentState(TypedDict, total=False):
    query: str

    plan: list[SubTask]
    plan_cursor: int

    retrieved_pages: Annotated[list[PageHit], add]
    extracted_facts: Annotated[list[Fact], add]
    computed_values: Annotated[list[ComputedValue], add]

    reflexion_iter: int
    is_sufficient: bool
    missing_info: str

    answer: str
    citations: list[Citation]

    chat_history: list[dict]
