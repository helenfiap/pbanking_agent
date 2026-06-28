import time

from agent.state import AgentState
from utils.llm import get_llm
from utils.llm_output import extract_text

SQL_PROMPT = """Você é um especialista em SQLite. Gere uma query SQL para responder a pergunta do usuário.

Schema do banco:
{schema}

Plano:
{plan}

Pergunta: {question}

Regras:
- Retorne APENAS a query SQL pura, sem markdown, sem explicação
- Use sintaxe SQLite (strftime para datas, não DATE_FORMAT)
- Trate valores NULL com COALESCE ou IS NOT NULL quando necessário
- Use sempre aliases de tabela para legibilidade"""


def sql_generator(state: AgentState) -> AgentState:
    """
    Node 3 — Heavy LLM.
    Translates the plan into executable SQL.
    Strips markdown artifacts from output before storing.
    """
    start = time.perf_counter()
    llm = get_llm("heavy")
    prompt = SQL_PROMPT.format(
        schema=state["schema"],
        plan=state["plan"],
        question=state["question"],
    )
    response = llm.invoke(prompt)
    sql = extract_text(response).strip().replace("```sql", "").replace("```", "").strip()
    elapsed = time.perf_counter() - start
    return {
        **state,
        "sql": sql,
        "error": None,
        "trace": state["trace"] + [f"✔ SQL gerado ({elapsed:.2f}s)"],
    }
