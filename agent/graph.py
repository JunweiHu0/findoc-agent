from langgraph.graph import END, StateGraph
from loguru import logger

from .config import MAX_REFLEXION_ITER
from .llm import has_llm_key
from .nodes.executor import executor_node
from .nodes.planner import planner_node
from .nodes.synthesizer import synthesizer_node
from .nodes.verifier import verifier_node
from .state import AgentState


def _route_after_verifier(state: AgentState) -> str:
    if state.get("is_sufficient", False):
        return "synthesizer"
    if state.get("reflexion_iter", 0) >= MAX_REFLEXION_ITER:
        logger.info(f"reflexion budget exhausted ({MAX_REFLEXION_ITER}); forcing synthesis")
        return "synthesizer"
    return "executor"


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("verifier", verifier_node)
    g.add_node("synthesizer", synthesizer_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "verifier")
    g.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"executor": "executor", "synthesizer": "synthesizer"},
    )
    g.add_edge("synthesizer", END)
    return g


def compile_graph():
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
    }
    out = app.invoke(init)
    print("\n=== Final state ===")
    for k, v in out.items():
        print(f"{k}: {v}")
