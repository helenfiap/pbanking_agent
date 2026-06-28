import time

from agent.state import AgentState
from utils.llm import get_llm
from utils.llm_output import extract_text

PLANNER_PROMPT = """Você é um analista de dados sênior de uma empresa fintech (marketplace de produtos financeiros).
Você tem acesso a um banco SQLite com dados de clientes, compras, suporte e campanhas de marketing.

Schema do banco:
{schema}

Pergunta do usuário: {question}

Escreva um plano conciso para responder essa pergunta com SQL. Inclua:
- Quais tabelas são necessárias e por quê
- Quais joins ou agregações são necessários
- Se uma query é suficiente ou se precisam de múltiplos passos

Responda sempre em português brasileiro. Seja direto — este plano guia o gerador de SQL."""


def planner(state: AgentState) -> AgentState:
    """
    Node 2 — Heavy LLM.
    Produces a reasoning plan before writing any SQL — reduces hallucinations.
    """
    start = time.perf_counter()
    llm = get_llm("heavy")
    prompt = PLANNER_PROMPT.format(schema=state["schema"], question=state["question"])
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start
    return {
        **state,
        "plan": extract_text(response),
        "trace": state["trace"] + [f"✔ Plano criado ({elapsed:.2f}s)"],
    }
