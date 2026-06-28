import time

from agent.state import AgentState
from utils.database import run_query


def sql_executor(state: AgentState) -> AgentState:
    """
    Node 4 — No LLM, fully deterministic.
    Runs the SQL against SQLite and catches any exception.
    On failure: stores the error message and increments the retry counter.
    """
    start = time.perf_counter()
    try:
        df = run_query(state["sql"])
        elapsed = time.perf_counter() - start
        return {
            **state,
            "result": df.to_json(orient="records"),
            "error": None,
            "trace": state["trace"] + [f"✔ Query executada — {len(df)} linhas ({elapsed:.2f}s)"],
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {
            **state,
            "result": None,
            "error": str(e),
            "retries": state["retries"] + 1,
            "trace": state["trace"] + [f"✘ Erro SQL (tentativa {state['retries'] + 1}, {elapsed:.2f}s): {str(e)[:80]}"],
        }
