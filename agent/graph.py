from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.schema_inspector import schema_inspector
from agent.nodes.planner import planner
from agent.nodes.sql_generator import sql_generator
from agent.nodes.sql_executor import sql_executor
from agent.nodes.error_recovery import error_recovery
from agent.nodes.visualization_agent import visualization_agent
from agent.nodes.response_formatter import response_formatter

MAX_RETRIES = 2


def route_after_executor(state: AgentState) -> str:
    """
    Conditional edge — the only branching logic in the graph.
    Called automatically by LangGraph after sql_executor runs.
    Returns the name of the next node to execute.
    """
    if state.get("error") and state.get("retries", 0) < MAX_RETRIES:
        return "error_recovery"
    elif state.get("error"):
        # Max retries exceeded: go straight to formatter with a graceful fail message
        return "response_formatter"
    else:
        return "visualization_agent"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("schema_inspector", schema_inspector)
    graph.add_node("planner", planner)
    graph.add_node("sql_generator", sql_generator)
    graph.add_node("sql_executor", sql_executor)
    graph.add_node("error_recovery", error_recovery)
    graph.add_node("visualization_agent", visualization_agent)
    graph.add_node("response_formatter", response_formatter)

    # Entry point
    graph.set_entry_point("schema_inspector")

    # Linear edges
    graph.add_edge("schema_inspector", "planner")
    graph.add_edge("planner", "sql_generator")
    graph.add_edge("sql_generator", "sql_executor")

    # Conditional branch after executor
    graph.add_conditional_edges(
        "sql_executor",
        route_after_executor,
        {
            "error_recovery": "error_recovery",
            "visualization_agent": "visualization_agent",
            "response_formatter": "response_formatter",
        },
    )

    # Error recovery loops back to sql_generator for a fresh attempt
    graph.add_edge("error_recovery", "sql_generator")

    # Happy path
    graph.add_edge("visualization_agent", "response_formatter")
    graph.add_edge("response_formatter", END)

    return graph.compile()


# Singleton — import this in app.py and any test
agent_graph = build_graph()
