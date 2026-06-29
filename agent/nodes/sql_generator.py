import time

from agent.state import AgentState
from utils.llm import get_llm
from utils.llm_output import extract_text

SQL_PROMPT = """Você é um especialista em SQLite. Gere uma query SQL para responder a pergunta do usuário.

Schema do banco (com valores exatos das colunas categóricas):
{schema}

Plano:
{plan}

Pergunta: {question}

Regras obrigatórias:
- Retorne APENAS a query SQL pura, sem markdown, sem explicação
- Use sintaxe SQLite (strftime para datas, nunca DATE_FORMAT)
- Use EXATAMENTE os valores de string listados no schema — SQLite é case-sensitive
  Exemplos corretos: canal = 'App', canal = 'WhatsApp', canal = 'E-mail'
  Exemplos ERRADOS:  canal = 'app', canal = 'whatsapp', canal = 'email'
- Para booleanos use 0/1: resolvido = 0, interagiu = 1
- Para filtrar por ano: strftime('%Y', coluna) = '2024'
- Para filtrar por mês: strftime('%m', coluna) = '05'
- NÃO use date('now') para filtros de "último ano" ou "ano passado" —
  os dados vão de 2024 a 2025, use o range real do schema
- Use aliases de tabela para legibilidade
- Trate NULL com COALESCE ou IS NOT NULL quando necessário
- SEMPRE use alias descritivos em portugues para colunas calculadas:
  COUNT(*) AS total, COUNT(DISTINCT x) AS total_x, AVG(x) AS media_x, SUM(x) AS total_x
  Nunca deixe colunas sem alias como "COUNT(DISTINCT cm.cliente_id)" — use AS total_clientes"""


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
