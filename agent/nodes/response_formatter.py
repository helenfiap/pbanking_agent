import time
from io import StringIO

import pandas as pd
from agent.state import AgentState
from utils.llm import get_llm
from utils.llm_output import extract_text

FORMATTER_PROMPT = """Você é um assistente analista de dados de uma empresa fintech.
Escreva uma resposta breve e direta para a pergunta do usuário com base nos resultados da query.

Pergunta: {question}
Resumo dos resultados: {result_summary}
Erro (se houver): {error}

Regras:
- Máximo 2-3 frases
- Seja específico: mencione números, nomes e valores dos dados
- Se houve erro após as tentativas, explique claramente o que não foi possível responder e por quê
- Responda sempre em português brasileiro"""


def response_formatter(state: AgentState) -> AgentState:
    """
    Node 7 — Cheap LLM. Final node before END.
    Produces the natural language answer shown in the chat.
    Also runs on the graceful-fail path (max retries exceeded).
    """
    start = time.perf_counter()
    result_summary = "Nenhum dado retornado."
    if state.get("result"):
        df = pd.read_json(StringIO(state["result"]), orient="records")
        if len(df) > 0:
            result_summary = (
                f"{len(df)} linhas retornadas.\n"
                f"Colunas: {list(df.columns)}\n"
                f"Primeiras linhas:\n{df.head(5).to_string(index=False)}"
            )

    llm = get_llm("cheap")
    prompt = FORMATTER_PROMPT.format(
        question=state["question"],
        result_summary=result_summary,
        error=state.get("error") or "nenhum",
    )
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start
    return {
        **state,
        "response": extract_text(response),
        "trace": state["trace"] + [f"✔ Resposta formatada ({elapsed:.2f}s)"],
    }
