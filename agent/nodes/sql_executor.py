import re
import time

from agent.state import AgentState
from utils.database import run_query

# ── SQL Safety ────────────────────────────────────────────────────────────────
_FORBIDDEN = re.compile(
    r'^\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|DETACH)\b',
    re.IGNORECASE | re.MULTILINE,
)

def _check_sql_safety(sql: str) -> str | None:
    """Returns error message if SQL contains forbidden statements, else None."""
    match = _FORBIDDEN.search(sql)
    if match:
        return f"SQL bloqueado por segurança: '{match.group().strip()}' não é permitido. Apenas SELECT é aceito."
    return None


# ── Column name cleanup ───────────────────────────────────────────────────────
_AGG_PATTERN = re.compile(
    r'^(COUNT|SUM|AVG|MIN|MAX|ROUND|COALESCE|CAST|LENGTH|LOWER|UPPER|TRIM|SUBSTR|STRFTIME)\s*\(',
    re.IGNORECASE,
)
_CLEAN_MAP = {
    "count": "total", "sum": "soma", "avg": "media",
    "min": "minimo", "max": "maximo",
}

def _clean_columns(df):
    new_cols = {}
    seen = {}
    for col in df.columns:
        m = _AGG_PATTERN.match(col.strip())
        if m:
            func = m.group(1).lower()
            base = _CLEAN_MAP.get(func, func)
            n = seen.get(base, 0)
            seen[base] = n + 1
            new_cols[col] = base if n == 0 else f"{base}_{n}"
    return df.rename(columns=new_cols) if new_cols else df


# ── Node ──────────────────────────────────────────────────────────────────────
def sql_executor(state: AgentState) -> AgentState:
    """
    Node 4 — No LLM, fully deterministic.
    1. Rejects any SQL that is not a SELECT (security guard).
    2. Runs the query against SQLite.
    3. Cleans up raw-expression column names.
    On failure: stores the error message and increments the retry counter.
    """
    start = time.perf_counter()

    safety_error = _check_sql_safety(state["sql"])
    if safety_error:
        elapsed = time.perf_counter() - start
        return {
            **state,
            "result": None,
            "error": safety_error,
            "retries": state["retries"] + 1,
            "trace": state["trace"] + [f"✘ SQL bloqueado por segurança ({elapsed:.2f}s)"],
        }

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
