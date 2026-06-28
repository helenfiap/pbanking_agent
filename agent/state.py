from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    """
    The single source of truth that flows through every node in the graph.
    Each node receives this dict and returns an updated version of it.
    """
    question: str           # the user's natural language question
    schema: str             # database schema discovered dynamically
    plan: str               # planner's reasoning about how to approach the query
    sql: str                # the generated (or corrected) SQL query
    result: Optional[str]   # query result serialized as JSON string (DataFrame -> JSON)
    error: Optional[str]    # error message if SQL execution failed
    retries: int            # how many times error recovery has run (max 2)
    viz_spec: dict          # visualization spec: {type, x, y, color, title}
    response: str           # final natural language response shown to the user
    trace: list[str]        # reasoning steps shown in the UI panel (e.g. "✔ SQL generated")
