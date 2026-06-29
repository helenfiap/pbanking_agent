import json
import time
from io import StringIO

import pandas as pd
from agent.state import AgentState
from utils.llm import get_llm
from utils.llm_output import extract_text

VIZ_PROMPT = """Você é um especialista em visualização de dados. Escolha o melhor gráfico para este resultado.

Pergunta: {question}
Dica de intenção da pergunta: {intent_hint}
Número de linhas: {n_rows}
Colunas disponíveis: {columns}
Colunas numéricas: {numeric_cols}
Colunas de data/tempo: {date_cols}
Amostra dos dados (primeiras 3 linhas):
{sample}

Retorne APENAS um objeto JSON válido com esta estrutura exata:
{{
  "type": "bar|line|table|pie",
  "x": "nome_da_coluna",
  "y": "nome_da_coluna",
  "color": null,
  "orientation": "h|v",
  "title": "título descritivo curto em português"
}}

Regras obrigatórias:
- "line" se há coluna de data/mês/ano OU a dica de intenção for "tendência"
- "bar" com orientation "h" (horizontal) se a intenção for "ranking" — barras horizontais facilitam leitura de nomes longos
- "bar" com orientation "v" (vertical) para comparações simples entre 2-6 categorias curtas
- "pie" APENAS se <= 6 categorias E a pergunta pede proporção/percentual
- "table" se > 15 linhas, múltiplas colunas sem hierarquia clara, ou dica for "lista"
- x e y devem ser nomes EXATOS de colunas dos dados
- Para "bar" horizontal (h): x recebe a coluna numérica, y recebe a coluna categórica
- Para "bar" vertical (v): x recebe a coluna categórica, y recebe a coluna numérica
- Nunca use "bar" com apenas 1 linha de resultado"""


# Keyword-based intent classifier — deterministic, no LLM cost
INTENT_HINTS = {
    "tendência":   ["tendência", "evolução", "histórico", "ao longo", "por mês", "por ano", "últim",
                    "a cada mês", "a cada mes", "mês a mês", "mes a mes", "mensal", "anual",
                    "a cada semana", "semana a semana", "semanal", "por semana",
                    "a cada dia", "dia a dia", "diário", "diario"],
    "lista":       ["liste", "quais são", "quais foram", "mostre", "exiba"],
    "ranking":     ["top", "maior", "menor", "mais", "menos", "ranking", "melhores", "piores"],
    "proporção":   ["proporção", "percentual", "%", "distribuição", "fatia"],
    "contagem":    ["quantos", "quantidade", "número de", "total de"],
    "comparação":  ["comparar", "versus", "vs", "diferença entre"],
}

# Date-like column name patterns
DATE_PATTERNS = ["data", "mes", "mês", "ano", "year", "month", "date", "periodo", "período"]


def classify_intent(question: str) -> str:
    q = question.lower()
    for intent, keywords in INTENT_HINTS.items():
        if any(k in q for k in keywords):
            return intent
    return "geral"


def has_date_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(p in c.lower() for p in DATE_PATTERNS)]


def visualization_agent(state: AgentState) -> AgentState:
    """
    Node 6 — Hybrid: deterministic pre-filter + cheap LLM for ambiguous cases.

    Pre-filter handles obvious cases without an LLM call:
    - No data          → table
    - Single value     → metric (shown as st.metric in the UI)
    - Tendência + data → line chart (ignores row limit — time series have many rows)
    - Too many rows    → table
    - Ranking/contagem → horizontal bar

    LLM handles everything else.
    """
    start = time.perf_counter()

    if not state.get("result"):
        elapsed = time.perf_counter() - start
        return {
            **state,
            "viz_spec": {"type": "table", "title": "Sem resultados"},
            "trace": state["trace"] + [f"✔ Visualização: tabela — sem dados ({elapsed:.2f}s)"],
        }

    df = pd.read_json(StringIO(state["result"]), orient="records")
    n_rows, n_cols = len(df), len(df.columns)

    # ── Deterministic pre-filters ─────────────────────────────────────────────
    # Single scalar result (e.g. "quantos clientes...?")
    if n_rows == 1 and n_cols == 1:
        val = df.iloc[0, 0]
        elapsed = time.perf_counter() - start
        return {
            **state,
            "viz_spec": {
                "type": "metric",
                "label": df.columns[0],
                "value": str(val),
                "title": state["question"][:60],
            },
            "trace": state["trace"] + [f"✔ Visualização: métrica ({elapsed:.2f}s)"],
        }

    # Single row, multiple cols → table
    if n_rows == 1:
        elapsed = time.perf_counter() - start
        return {
            **state,
            "viz_spec": {"type": "table", "title": "Resultado"},
            "trace": state["trace"] + [f"✔ Visualização: tabela — linha única ({elapsed:.2f}s)"],
        }

    intent = classify_intent(state["question"])

    # Coerce potential numeric columns (handles JSON→pandas object dtype)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    numeric_cols_det = df.select_dtypes(include="number").columns.tolist()
    category_cols_det = df.select_dtypes(exclude="number").columns.tolist()
    date_cols_det = has_date_columns(df)

    # ── Tendência → line chart (ignora n_rows — séries temporais têm muitas linhas)
    if intent == "tendência" and date_cols_det and numeric_cols_det and n_rows <= 200:
        cat_non_date = [c for c in category_cols_det if c not in date_cols_det]
        color_col = cat_non_date[0] if cat_non_date else None
        elapsed = time.perf_counter() - start
        return {
            **state,
            "viz_spec": {
                "type": "line",
                "x": date_cols_det[0],
                "y": numeric_cols_det[0],
                "orientation": "v",
                "color": color_col,
                "title": state["question"][:60],
            },
            "trace": state["trace"] + [f"✔ Visualização: line — tendência temporal ({elapsed:.2f}s)"],
        }

    # ── Too many rows → table (exceto tendência, já tratada acima)
    if n_rows > 15:
        elapsed = time.perf_counter() - start
        return {
            **state,
            "viz_spec": {"type": "table", "title": "Resultados"},
            "trace": state["trace"] + [f"✔ Visualização: tabela — {n_rows} linhas ({elapsed:.2f}s)"],
        }

    # ── Ranking/contagem/comparação → horizontal bar
    if intent in ("ranking", "contagem", "comparação") and 2 <= n_rows <= 15:
        if len(numeric_cols_det) >= 1 and len(category_cols_det) >= 1:
            elapsed = time.perf_counter() - start
            return {
                **state,
                "viz_spec": {
                    "type": "bar",
                    "x": numeric_cols_det[0],
                    "y": category_cols_det[0],
                    "orientation": "h",
                    "color": None,
                    "title": state["question"][:60],
                },
                "trace": state["trace"] + [f"✔ Visualização: bar horizontal — intenção: {intent} ({elapsed:.2f}s)"],
            }

    # ── LLM for everything else ───────────────────────────────────────────────
    numeric_cols = list(df.select_dtypes(include="number").columns)
    date_cols = has_date_columns(df)

    llm = get_llm("cheap")
    prompt = VIZ_PROMPT.format(
        question=state["question"],
        intent_hint=intent,
        n_rows=n_rows,
        columns=list(df.columns),
        numeric_cols=numeric_cols,
        date_cols=date_cols,
        sample=df.head(3).to_string(index=False),
    )
    response = llm.invoke(prompt)

    try:
        content = extract_text(response).strip().replace("```json", "").replace("```", "").strip()
        viz_spec = json.loads(content)
    except json.JSONDecodeError:
        viz_spec = {"type": "table", "title": "Resultados"}

    elapsed = time.perf_counter() - start
    return {
        **state,
        "viz_spec": viz_spec,
        "trace": state["trace"] + [f"✔ Visualização: {viz_spec.get('type', 'tabela')} — intenção: {intent} ({elapsed:.2f}s)"],
    }
