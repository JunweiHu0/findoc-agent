"""Query router — decides whether the query needs document retrieval at all.
查询路由——判断查询是否需要启用知识库检索。

Direct-answer cases (no retrieval): greetings, general knowledge, follow-up
questions answerable from chat_history alone. Retrieval cases: anything that
requires looking up specific numbers/facts in indexed financial documents.
直接回答场景：闲聊、常识、基于历史对话的指代追问；检索场景：需要查文档具体数值/事实。
"""

from __future__ import annotations

from loguru import logger

from ..llm import get_llm, has_llm_key
from ..state import AgentState


# Strong-signal keywords — if present, almost certainly needs retrieval.
# 强信号关键词——出现这些几乎必然需要检索
_RETRIEVAL_KEYWORDS = [
    # Reporting periods / 报告期
    "年报", "季报", "半年报", "中报", "一季度", "二季度", "三季度", "四季度",
    # Financial metrics / 财务指标
    "营收", "营业收入", "净利润", "毛利率", "净利率", "ROE", "EPS", "市盈率", "市净率",
    "总资产", "负债", "现金流", "研发投入", "员工总数", "员工数", "净资产", "存货",
    # Financial statement structure / 报表结构
    "资产负债表", "利润表", "现金流量表", "附注", "审计意见",
    # Year markers (very strong) / 年份指标（很强）
    "2020年", "2021年", "2022年", "2023年", "2024年", "2025年",
]

# Direct-answer keywords — strong signal we DON'T need retrieval.
# 直接回答关键词——强烈暗示不需要检索
_DIRECT_ANSWER_KEYWORDS = [
    "你好", "您好", "谢谢", "感谢", "再见", "拜拜",
    "你是谁", "你能做什么", "你叫什么", "介绍一下你",
    "怎么用", "如何使用", "有什么功能",
    "什么是", "什么叫", "解释一下",  # generic concept Q
]


def _heuristic_decide(query: str) -> bool | None:
    """Fast keyword heuristic. Returns True/False, or None if ambiguous.
    关键词启发式：返回 True/False，无法判定时返回 None。"""
    q = query.strip()
    if not q:
        return False  # empty → no retrieval
    if len(q) <= 3:
        # Very short — likely greeting / 很短 → 多半是问候
        return False
    for kw in _RETRIEVAL_KEYWORDS:
        if kw in q:
            return True
    for kw in _DIRECT_ANSWER_KEYWORDS:
        if kw in q:
            return False
    return None


def _llm_decide(query: str, chat_history: list[dict]) -> bool:
    """Lightweight LLM call (~80 tokens) for ambiguous queries / 模糊查询走轻量 LLM 判断。"""
    if not has_llm_key():
        # No LLM available — default to retrieval (safer than guessing wrong).
        # 没 LLM 时默认走检索（误判检索成本远低于误判直答）
        return True

    history_str = "(无)"
    if chat_history:
        last = chat_history[-2:]
        history_str = " | ".join(
            f"{t.get('role','?')}:{(t.get('content') or '')[:60]}" for t in last
        )

    try:
        llm = get_llm("planner")
        prompt = (
            "判断下面的用户问题是否需要查阅金融年报/财报文档来回答。\n"
            "需要检索：涉及具体公司具体年份的财务数字、披露内容、报表项目等。\n"
            "无需检索：闲聊、问候、对前文的指代追问（无新事实查询）、解释通用概念、"
            "数学计算（不依赖文档数据）、Agent 自身能力问题。\n\n"
            f"对话历史: {history_str}\n"
            f"问题: {query}\n\n"
            "只输出一个词：retrieve 或 direct"
        )
        resp = llm.invoke(prompt)
        text = (resp.content or "").strip().lower()
        if "direct" in text:
            return False
        return True
    except Exception as e:
        logger.debug(f"query_router LLM fallback failed ({e}); defaulting to retrieval")
        return True


def query_router_node(state: AgentState) -> dict:
    """Decide whether retrieval is needed for this query / 判断本轮是否需要启用检索。

    Sets state["needs_retrieval"]. Downstream conditional edge routes
    True → retrieval_scout (full pipeline), False → synthesizer (direct answer).
    设置 needs_retrieval；图条件边据此路由检索 vs 直答。"""
    query = state.get("query", "")
    history = state.get("chat_history") or []

    decision = _heuristic_decide(query)
    if decision is None:
        decision = _llm_decide(query, history)
        source = "llm"
    else:
        source = "heuristic"

    logger.info(f"query_router: needs_retrieval={decision} (via {source})")
    return {"needs_retrieval": decision}
