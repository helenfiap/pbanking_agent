import time

from agent.state import AgentState
from utils.llm import get_llm
from utils.llm_output import extract_text

RECOVERY_PROMPT = """Você é um especialista em SQLite. Corrija a query SQL abaixo.

Schema:
{schema}

SQL com erro:
{sql}

Mensagem de erro:
{error}

Retorne APENAS a query SQL corrigida. Sem explicação, sem markdown."""


def error_recovery(state: AgentState) -> AgentState:
    """
    Node 5 — Cheap LLM.
    Receives bad SQL + exact error message and rewrites the query.
    Uses cheap model: task is focused — fix a specific syntax/logic error.
    """
    start = time.perf_counter()
    llm = get_llm("cheap")
    prompt = RECOVERY_PROMPT.format(
        schema=state["schema"],
        sql=state["sql"],
        error=state["error"],
    )
    response = llm.invoke(prompt)
    fixed_sql = extract_text(response).strip().replace("```sql", "").replace("```", "").strip()
    elapsed = time.perf_counter() - start
    return {
        **state,
        "sql": fixed_sql,
        "error": None,
        "trace": state["trace"] + [f"✔ SQL corrigido — tentativa {state['retries']} ({elapsed:.2f}s)"],
    }
