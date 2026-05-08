"""Agent shared state — TypedDict flowing through all LangGraph nodes / Agent 共享状态。

Fields annotated `Annotated[T, add]` use LangGraph's reducer pattern: each node
returns a delta and the runtime appends to the list. Annotated[T, add] 标注的字段
使用 LangGraph reducer 模式——节点返回增量，运行时自动追加而非覆盖。
"""

from operator import add
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class ErrorLogEntry(BaseModel):
    """Structured error record from any node or tool / 节点/工具产生的结构化错误记录。"""
    node: str = ""
    tool: str = ""
    error_type: str = ""
    message: str = ""
    retryable: bool = False
    fatal: bool = False
    timestamp: float = 0.0


class TodoItem(BaseModel):
    """Runtime task tracker, one per plan step / 运行时任务追踪，每个 plan 步骤一个。"""
    id: str = Field(default_factory=lambda: f"t-{__import__('uuid').uuid4().hex[:6]}")
    sub_task_idx: int = 0
    title: str = ""
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    attempt: int = 0
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    parent_id: Optional[str] = None     # links to original todo on retry / 重试时指向原始 todo


class SubTask(BaseModel):
    """One step in the planner's execution plan / Planner 执行计划中的一步。"""
    sub_query: str
    target_doc: Optional[str] = None
    expected_output_schema: str = "text"
    tool_calls: list[dict] = []         # explicit tool dispatch [{tool, args}] / 显式工具调用
    # DAG scheduling fields / DAG 调度字段
    task_id: str = ""
    depends_on: list[str] = []
    priority: Literal[0, 1] = 0         # 0=core 核心, 1=exploratory 探索
    status: Literal["pending", "running", "done", "failed"] = "pending"


class PageHit(BaseModel):
    """Page retrieved by ColQwen multi-vector MaxSim search / ColQwen 多向量 MaxSim 检索结果。"""
    doc_id: str
    page_num: int
    score: float = 0.0
    image_path: Optional[str] = None


class Fact(BaseModel):
    """Structured fact extracted from a page by VLM + fact_extractor / VLM+fact_extractor 抽取的结构化事实。"""
    text: str
    source_doc: str
    source_page: int
    sub_task_idx: Optional[int] = None
    # Structured extraction fields / 结构化抽取字段
    entity: Optional[str] = None        # company / 公司名
    period: Optional[str] = None        # "2023" / "2023Q1" / "2023H1"
    metric: Optional[str] = None        # "营业收入" / "毛利率"
    value: Optional[float] = None
    unit: Optional[str] = None          # "亿元" / "%"
    raw_kind: Literal["numeric", "string", "table_row", "unstructured"] = "unstructured"


class ComputedValue(BaseModel):
    """Numeric result from the AST-safe calculator / AST 安全计算器产出的数值结果。"""
    expr: str
    value: float
    sub_task_idx: Optional[int] = None


class Citation(BaseModel):
    """A (doc_id, page_num) pair referenced in the final answer / 最终答案中引用的 (doc_id, page_num)。"""
    doc_id: str
    page_num: int


class AgentState(TypedDict, total=False):
    """All state shared across the 8-node LangGraph pipeline / 8 节点 LangGraph 管线共享的全部状态。"""
    query: str

    plan: list[SubTask]
    plan_cursor: int

    # Accumulated evidence / 累积证据
    retrieved_pages: Annotated[list[PageHit], add]
    extracted_facts: Annotated[list[Fact], add]
    computed_values: Annotated[list[ComputedValue], add]

    # Reflexion control / 反射循环控制
    reflexion_iter: int
    is_sufficient: bool
    missing_info: str                              # deprecated free-text, kept for compat / 已弃用自由文本，保留兼容
    missing_facts: Annotated[list[dict], add]       # structured [{sub_task_idx, what, root_cause, ...}] / 结构化缺失信息
    tried_queries: list[str]                         # full deduped snapshot, overwritten by verifier each iter / 由 verifier 每轮覆盖写入的去重快照
    tried_pages: list[dict]                          # full deduped snapshot: [{doc_id, page_num}] / 由 verifier 每轮覆盖写入的去重快照
    budget_retrievals: int                           # remaining retrieval budget / 剩余检索预算
    budget_vlm_calls: int                            # remaining VLM call budget / 剩余 VLM 调用预算
    remediation_hint: Optional[dict]                # strategy hints carried into executor / 注入 executor 的修复策略

    # Pre-retrieval scout / 检索前探查
    scout_candidates: list[dict]                    # [{doc_id, top_page_num, top_score}]

    # Post-hoc grounding audit / 后置事实校验
    unverified_claims: Annotated[list[dict], add]
    grounding_score: float

    # Memory / 记忆
    fact_index: dict                                # {(entity, period, metric): Fact} working memory / 工作记忆
    known_facts: list[dict]                         # cross-turn cached facts / 跨轮缓存事实

    # Observability / 可观测性
    error_log: Annotated[list[dict], add]           # structured error records / 结构化错误记录
    todo_items: Annotated[list[dict], add]           # runtime task status / 运行时任务状态
    todo_updates: Annotated[list[dict], add]         # incremental todo updates for SSE / SSE 增量状态更新

    # Dynamic prompt routing / 动态 prompt 路由
    query_class: str                                # "single_fact" | "cross_doc_compare" | ...
    agent_profile: dict                             # specialist routing config (reserved) / specialist 路由配置（预留）

    # plan_critic re-entry guard / plan_critic 重入保护
    plan_critic_last_cursor: int                    # last plan_cursor at which plan_critic ran (-1=never) / 上次 plan_critic 运行时的 plan_cursor
    plan_critic_iter: int                           # times plan_critic has revised the plan / plan_critic 已修订次数

    # Routing decision — set by query_router_node, consumed by graph conditional edge.
    # 路由决策——query_router 设置，图条件边消费
    needs_retrieval: bool

    # Final output / 最终输出
    answer: str
    citations: list[Citation]

    # Multi-turn / 多轮对话
    chat_history: list[dict]
    doc_filter: Optional[list[str]]                 # user-uploaded document scope / 用户上传文档过滤
