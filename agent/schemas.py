"""LLM structured-output schemas — pydantic models for json_mode generation.

Passed to ChatOpenAI.with_structured_output() to force the LLM to emit valid
JSON matching these shapes. 传给 ChatOpenAI.with_structured_output() 的 pydantic
模型，强制 LLM 产出符合这些结构体的 JSON。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation planned by the planner / Planner 规划的单次工具调用。"""
    tool: str = Field(description="Tool name as listed in the Available Tools registry / 工具名称")
    args: dict = Field(default_factory=dict, description="Keyword arguments matching the tool's params_schema / 工具参数")


class PlanItemSchema(BaseModel):
    """One step in the planner's execution plan, DAG-aware / Planner 执行计划中的一步，支持 DAG 依赖。"""
    sub_query: str = Field(description="Retrieval query or arithmetic expression / 检索词或算术表达式")
    target_doc: Optional[str] = Field(default=None, description="Restrict retrieval to this doc_id / 限定检索文档")
    expected_output_schema: str = Field(default="text", description='"number" | "text" | "table"')
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Explicit tool dispatch list / 显式工具调度列表")
    task_id: str = Field(default="", description="Unique id for DAG scheduling / DAG 调度唯一标识")
    depends_on: list[str] = Field(default_factory=list, description="Task IDs this step must wait for / 依赖的前置步骤 ID")
    priority: Literal[0, 1] = Field(default=0, description="0=core 核心, 1=exploratory 探索")


class PlannerOutput(BaseModel):
    """Structured output from the planner LLM call / Planner LLM 调用的结构化输出。"""
    plan: list[PlanItemSchema] = Field(default_factory=list, description="Ordered list of sub-tasks (DAG-aware) / 子任务列表，支持 DAG 依赖")
    query_class: str = Field(default="", description='Detected query class: single_fact | cross_doc_compare | multi_step_calc | trend_analysis')


class TodoItemSchema(BaseModel):
    """Schema for runtime todo tracking / 运行时任务追踪 schema。"""
    id: str = Field(default="")
    sub_task_idx: int = Field(default=0)
    title: str = Field(default="")
    status: Literal["pending", "running", "done", "failed", "skipped"] = Field(default="pending")
    attempt: int = Field(default=0)
    error: Optional[str] = Field(default=None)
    parent_id: Optional[str] = Field(default=None)


# ---------------------------------------------------------------------------
# Verifier structured output / Verifier 结构化输出
# ---------------------------------------------------------------------------

class MissingFact(BaseModel):
    """A gap diagnosed by the verifier, with root cause and suggested remedy / Verifier 诊断的缺失事实，含根因和修复建议。"""
    sub_task_idx: int = Field(description="Index (0-based) of the sub-task that left a gap / 产生缺失的子任务索引")
    what: str = Field(description="Human-readable description of what is missing / 缺失内容的可读描述")
    root_cause: Literal[
        "retrieval_miss",           # ColQwen did not recall the right page / 检索未召回正确页面
        "reading_miss",             # correct page recalled but VLM missed the figure / 页面正确但 VLM 漏读
        "ambiguous_query",          # the sub_query itself was ambiguous / 子查询本身有歧义
        "inconsistency",            # cross-page numbers conflict / 跨页数字冲突
    ] = Field(description="Root cause category — drives downstream remediation strategy / 根因类别，驱动下游修复策略")
    suggested_query: Optional[str] = Field(default=None, description="Rewritten query for re-retrieval / 重检索的改写查询")
    suggested_target_doc: Optional[str] = Field(default=None, description="Constrain re-retrieval to this doc_id / 限定重检索的文档")
    suggested_page_nums: Optional[list[int]] = Field(default=None, description="Re-read only these page numbers / 仅重读这些页码")


class VerifierOutput(BaseModel):
    """Structured output from the verifier LLM call / Verifier LLM 调用的结构化输出。"""
    is_sufficient: bool = Field(description="True iff collected evidence is enough to answer the question / 证据是否足够回答")
    inconsistency: str = Field(default="", description="Non-empty if numbers/calibers conflict across pages / 跨页数字/口径冲突描述")
    missing_facts: list[MissingFact] = Field(default_factory=list, description="Structured list of what is missing and why / 结构化缺失列表")
    missing_info: str = Field(default="", description="Deprecated free-text fallback; prefer missing_facts / 已弃用自由文本回退，优先用 missing_facts")
