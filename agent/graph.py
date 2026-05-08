"""LangGraph state-machine assembly — FinDoc agent topology.

Topology / 拓扑:
    query_router → [needs_retrieval?]
                   ├─ False → synthesizer (direct answer) → END
                   └─ True  → retrieval_scout → planner → executor → plan_critic? →
                              executor/verifier → (sufficient?) → synthesizer/remediation → executor
                              → synthesizer → END

Grounding node was removed (regex+if/else logic was brittle and added little value);
citation extraction now happens inside synthesizer by parsing [doc_id p.N] from the
generated answer. 删除了 grounding 节点，引用解析下放到 synthesizer。
"""

from langgraph.graph import END, StateGraph
from loguru import logger

from .config import MAX_REFLEXION_ITER
from .llm import has_llm_key
from .nodes.executor import executor_node
from .nodes.plan_critic import plan_critic_node, should_trigger_plan_critic
from .nodes.planner import planner_node, retrieval_scout_node
from .nodes.query_router import query_router_node
from .nodes.remediation import remediation_node
from .nodes.synthesizer import synthesizer_node
from .nodes.verifier import verifier_node
from .state import AgentState


def _route_after_verifier(state: AgentState) -> str:
    """Conditional edge: route to synthesizer if evidence is sufficient or budget exhausted, else remediation."""
    if state.get("is_sufficient", False):
        return "synthesizer"
    if state.get("reflexion_iter", 0) >= MAX_REFLEXION_ITER:
        logger.info(f"reflexion budget exhausted ({MAX_REFLEXION_ITER}); forcing synthesis")
        return "synthesizer"
    return "remediation"


def _route_after_executor(state: AgentState) -> str:
    """Conditionally route to plan_critic if signal detected, otherwise verifier.
    检测到信号时路由到 plan_critic，否则到 verifier。"""
    if should_trigger_plan_critic(state):
        logger.info("plan_critic triggered — routing to plan review")
        return "plan_critic"
    return "verifier"


def _route_after_query_router(state: AgentState) -> str:
    """Skip retrieval pipeline entirely when the router says the query is directly answerable.
    路由器判定可直答时，跳过整条检索流水线。"""
    return "retrieval_scout" if state.get("needs_retrieval", True) else "synthesizer"


def build_graph() -> StateGraph:
    """Construct the raw StateGraph with all nodes and edges (not yet compiled)."""
    g = StateGraph(AgentState)
    g.add_node("query_router", query_router_node)
    g.add_node("retrieval_scout", retrieval_scout_node)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("plan_critic", plan_critic_node)
    g.add_node("verifier", verifier_node)
    g.add_node("remediation", remediation_node)
    g.add_node("synthesizer", synthesizer_node)

    g.set_entry_point("query_router")
    g.add_conditional_edges(
        "query_router",
        _route_after_query_router,
        {"retrieval_scout": "retrieval_scout", "synthesizer": "synthesizer"},
    )
    g.add_edge("retrieval_scout", "planner")
    g.add_edge("planner", "executor")
    g.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"verifier": "verifier", "plan_critic": "plan_critic"},
    )
    g.add_edge("plan_critic", "executor")  # loop back after revision
    g.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"remediation": "remediation", "synthesizer": "synthesizer"},
    )
    g.add_edge("remediation", "executor")
    g.add_edge("synthesizer", END)
    return g


def compile_graph():
    """Return a compiled LangGraph runnable, ready for stream() / invoke()."""
    return build_graph().compile()


if __name__ == "__main__":
    logger.info(f"DEEPSEEK_API_KEY present: {has_llm_key()}")
    app = compile_graph()
    init: AgentState = {
        "query": "对比贵州茅台和宁德时代 2023 年的毛利率",
        "plan_cursor": 0,
        "reflexion_iter": 0,
        "is_sufficient": False,
        "retrieved_pages": [],
        "extracted_facts": [],
        "computed_values": [],
        "tried_queries": [],
        "tried_pages": [],
        "missing_facts": [],
        "budget_retrievals": 10,
        "budget_vlm_calls": 20,
        "scout_candidates": [],
        "unverified_claims": [],
        "fact_index": {},
        "known_facts": [],
        "error_log": [],
        "todo_items": [],
        "todo_updates": [],
        "query_class": "",
        "agent_profile": {},
        "plan_critic_last_cursor": -1,
        "plan_critic_iter": 0,
        "needs_retrieval": True,
    }
    out = app.invoke(init)
    print("\n=== Final state ===")
    for k, v in out.items():
        print(f"{k}: {v}")
