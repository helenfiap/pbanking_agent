"""
Minimal test suite — 3 tests covering critical agent behaviour.
Run with: uv run pytest tests/
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Test 1: Intent classifier ─────────────────────────────────────────────────
def test_classify_intent_tendencia():
    # classify_intent has no LLM dependency — import directly
    with patch.dict("sys.modules", {
        "langchain_openai": MagicMock(),
        "langchain_google_genai": MagicMock(),
        "utils.llm": MagicMock(get_llm=MagicMock()),
        "utils.llm_output": MagicMock(extract_text=MagicMock()),
    }):
        from agent.nodes.visualization_agent import classify_intent
        assert classify_intent("Qual a tendência de reclamações por canal no último ano?") == "tendência"
        assert classify_intent("evolução mensal de compras") == "tendência"
        assert classify_intent("quantos clientes interagiram?") == "contagem"
        assert classify_intent("top 5 estados") == "ranking"


# ── Test 2: Viz agent returns line + color for mes+canal+total data ───────────
def test_viz_agent_tendencia_multiline():
    with patch.dict("sys.modules", {
        "langchain_openai": MagicMock(),
        "langchain_google_genai": MagicMock(),
        "utils.llm": MagicMock(get_llm=MagicMock()),
        "utils.llm_output": MagicMock(extract_text=MagicMock()),
    }):
        from agent.nodes.visualization_agent import visualization_agent

        rows = [
            {"mes": "2024-07", "canal": "Chat",     "total_reclamacoes": 2},
            {"mes": "2024-07", "canal": "E-mail",   "total_reclamacoes": 1},
            {"mes": "2024-07", "canal": "Telefone", "total_reclamacoes": 3},
            {"mes": "2024-08", "canal": "Chat",     "total_reclamacoes": 3},
            {"mes": "2024-08", "canal": "E-mail",   "total_reclamacoes": 2},
            {"mes": "2024-08", "canal": "Telefone", "total_reclamacoes": 2},
        ]
        state = {
            "question": "Qual a tendência de reclamações por canal no último ano?",
            "result": json.dumps(rows),
            "viz_spec": {},
            "trace": [],
        }
        result = visualization_agent(state)
        viz = result["viz_spec"]

        assert viz["type"] == "line",    f"Expected 'line', got '{viz['type']}'"
        assert viz["x"] == "mes",       f"Expected x='mes', got x='{viz.get('x')}'"
        assert viz["color"] == "canal", f"Expected color='canal', got color='{viz.get('color')}'"


# ── Test 3: SQL executor blocks destructive statements ────────────────────────
def test_sql_executor_blocks_dangerous_sql():
    from agent.nodes.sql_executor import _check_sql_safety

    assert _check_sql_safety("SELECT * FROM clientes") is None
    assert _check_sql_safety("DROP TABLE clientes") is not None
    assert _check_sql_safety("DELETE FROM compras WHERE 1=1") is not None
    assert _check_sql_safety("UPDATE clientes SET nome='hack'") is not None
    assert _check_sql_safety("INSERT INTO clientes VALUES (1,'x')") is not None
    assert _check_sql_safety("ALTER TABLE clientes ADD COLUMN x TEXT") is not None
    assert _check_sql_safety("drop table clientes") is not None
    assert _check_sql_safety("  DELETE FROM suporte") is not None
