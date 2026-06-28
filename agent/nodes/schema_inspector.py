import time

from agent.state import AgentState
from utils.database import get_schema


def schema_inspector(state: AgentState) -> AgentState:
    """
    Node 1 — No LLM, fully deterministic.
    Reads the SQLite schema at runtime so the agent never depends on hardcoded table names.
    """
    start = time.perf_counter()
    schema = get_schema()
    elapsed = time.perf_counter() - start
    return {
        **state,
        "schema": schema,
        "trace": state["trace"] + [f"✔ Schema descoberto ({elapsed:.2f}s)"],
    }
