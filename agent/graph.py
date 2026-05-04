"""LangGraph state-machine assembly — 7-node topology for the FinDoc agent.

Nodes:
    retrieval_scout -> planner -> executor -> verifier ->
      sufficient -> synthesizer -> grounding -> END
      not sufficient -> remediation -> executor (loop, max MAX_REFLEXION_ITER)

The graph itself is stateless; AgentState flows through nodes as a TypedDict.
"""

from langgraph.graph import END, StateGraph
from loguru import logger

from .config import MAX_REFLEXION_ITER
from .llm import has_llm_key
from .nodes.executor import executor_node
from .nodes.grounding import grounding_node
from .nodes.planner import planner_node, retrieval_scout_node
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


def build_graph() -> StateGraph:
    """Construct the raw StateGraph with all 7 nodes and edges (not yet compiled)."""
    g = StateGraph(AgentState)
    g.add_node("retrieval_scout", retrieval_scout_node)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("verifier", verifier_node)
    g.add_node("remediation", remediation_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("grounding", grounding_node)

    g.set_entry_point("retrieval_scout")
    g.add_edge("retrieval_scout", "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "verifier")
    g.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"remediation": "remediation", "synthesizer": "synthesizer"},
    )
    g.add_edge("remediation", "executor")
    g.add_edge("synthesizer", "grounding")
    g.add_edge("grounding", END)
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
    }
    out = app.invoke(init)
    print("\n=== Final state ===")
    for k, v in out.items():
        print(f"{k}: {v}")
