import re
import time

from agent.state import AgentState
from utils.database import run_query

# Patterns that produce ugly column names when not aliased
_AGG_PATTERN = re.compile(
    r'^(COUNT|SUM|AVG|MIN|MAX|ROUND|COALESCE|CAST|LENGTH|LOWER|UPPER|TRIM|SUBSTR|STRFTIME)\s*\(',
    re.IGNORECASE,
)

_CLEAN_MAP = {
    "count": "total",
    "sum":   "soma",
    "avg":   "media",
    "min":   "minimo",
    "max":   "maximo",
}


def _clean_columns(df):
    """
    Rename any column that still looks like a raw SQL expression.
    e.g. COUNT(DISTINCT cm.cliente_id) -> total
         AVG(valor_total_gasto)         -> media
    """
    new_cols = {}
    seen = {}
    for col in df.columns:
        m = _AGG_PATTERN.match(col.strip())
        if m:
            func = m.group(1).lower()
            base = _CLEAN_MAP.get(func, func)
            # avoid duplicate names
            n = seen.get(base, 0)
            seen[base] = n + 1
            new_cols[col] = base if n == 0 else f"{base}_{n}"
    return df.rename(columns=new_cols) if new_cols else df


def sql_executor(state: AgentState) -> AgentState:
    """
    Node 4 — No LLM, fully deterministic.
    Runs the SQL against SQLite, cleans up raw-expression column names,
    and catches any exception.
    On failure: stores the error message and increments the retry counter.
    """
    start = time.perf_counter()
    try:
        df = run_query(state["sql"])
        df = _clean_columns(df)
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
