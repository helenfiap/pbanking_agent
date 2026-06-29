import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO

# load_dotenv FIRST — before any agent/utils imports so .env keys are available
# when llm.py is imported (avoids empty Gemini key list)
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import plotly.express as px
import streamlit as st

from agent.graph import agent_graph
from agent.nodes.visualization_agent import classify_intent
from agent.state import AgentState
try:
    from langfuse.callback import CallbackHandler as LangfuseCallback
except ImportError:
    try:
        from langfuse.langchain import CallbackHandler as LangfuseCallback
    except ImportError:
        LangfuseCallback = None

from utils.database import get_schema
from utils.logger import get_banker_metrics, log_query, estimate_cost_usd
from utils.llm import _thread_local as _llm_thread_local


def get_langfuse_handler(session_id: str = None):
    if LangfuseCallback is None:
        return None
    try:
        return LangfuseCallback(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_BASE_URL"),
            session_id=session_id,
        )
    except Exception:
        return None  # LangFuse optional — never crash the agent


# ── Helpers ───────────────────────────────────────────────────────────────────
def enrich_trace_with_pct(trace: list[str]) -> list[str]:
    times = []
    for step in trace:
        match = re.search(r'\((\d+\.\d+)s', step)
        times.append(float(match.group(1)) if match else 0.0)
    total = sum(times)
    if total == 0:
        return trace
    enriched = []
    for step, t in zip(trace, times):
        pct = t / total * 100
        enriched.append(re.sub(r'\((\d+\.\d+)s', f'({t:.2f}s — {pct:.0f}%', step))
    return enriched


def total_latency(trace: list[str]) -> float:
    total = 0.0
    for step in trace:
        match = re.search(r'\((\d+\.\d+)s', step)
        if match:
            total += float(match.group(1))
    return round(total, 2)


MODEL_TIMEOUT_S = 45  # max seconds to wait per model in benchmark


def run_agent(question: str, session_id: str = None) -> dict:
    initial_state: AgentState = {
        "question": question, "schema": "", "plan": "", "sql": "",
        "result": None, "error": None, "retries": 0,
        "viz_spec": {}, "response": "", "trace": [],
    }
    handler = get_langfuse_handler(session_id)
    config = {"callbacks": [handler]} if handler else {}
    return agent_graph.invoke(initial_state, config=config)


def run_agent_for_provider(provider: str, question: str) -> dict:
    """
    Run agent with a specific provider — safe for ThreadPoolExecutor.

    Uses thread-local storage instead of os.environ to avoid race conditions
    when multiple threads run simultaneously in the benchmark tab.
    """
    _llm_thread_local.provider = provider  # thread-local — no race condition
    try:
        state = run_agent(question, session_id=f"benchmark:{provider}")
        return {
            "success":  True,
            "latency":  total_latency(state["trace"]),
            "retries":  state.get("retries", 0),
            "sql":      state.get("sql", ""),
            "response": state.get("response", ""),
            "trace":    state.get("trace", []),
            "viz_spec": state.get("viz_spec", {}),
            "result":   state.get("result"),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "latency": 0}


def render_viz(final_state: dict, key: str = ""):
    if not final_state.get("result") or not final_state.get("viz_spec"):
        return
    df = pd.read_json(StringIO(final_state["result"]), orient="records")
    viz = final_state["viz_spec"]
    chart_type = viz.get("type", "table")
    x, y = viz.get("x"), viz.get("y")
    title = viz.get("title", "")
    color = viz.get("color") if viz.get("color") in df.columns else None
    orientation = viz.get("orientation", "v")

    if chart_type == "metric":
        st.metric(label=viz.get("label", "Resultado"), value=viz.get("value", ""))
    elif chart_type == "bar" and x and y and x in df.columns and y in df.columns:
        fig = px.bar(df, x=x, y=y, title=title, color=color, orientation=orientation)
        if orientation == "h":
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True, key=f"viz_bar_{key}")
    elif chart_type == "line" and x and y and x in df.columns and y in df.columns:
        # Safety: if x is not a date/time column, a line over categories is meaningless → bar
        x_is_date = any(p in x.lower() for p in ["mes", "data", "ano", "year", "month", "date", "periodo"])
        if not x_is_date:
            fig = px.bar(df, x=y, y=x, title=title, color=color, orientation="h")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True, key=f"viz_bar_{key}_fallback")
        else:
            fig = px.line(df, x=x, y=y, title=title, markers=True, color=color)
            st.plotly_chart(fig, use_container_width=True, key=f"viz_line_{key}")
    elif chart_type == "pie" and x and y and x in df.columns and y in df.columns:
        fig = px.pie(df, names=x, values=y, title=title)
        st.plotly_chart(fig, use_container_width=True, key=f"viz_pie_{key}")
    else:
        st.dataframe(df, width="stretch")



PROVIDER_LABELS = {
    "gemini":         "⚡ Gemini 2.5 Flash",
    "gemini-35":      "🌿 Gemini 3.5 Flash",
    "gemini-pro":     "✨ Gemini 2.5 Pro",
    "azure-deepseek": "🐋 DeepSeek V3.2",
    "azure":          "🔵 GPT-4o",
    "azure-41":       "🟣 GPT-4.1",
    "azure-kimi":     "🌙 Kimi K2.5",
    "azure-grok":     "⚡ Grok 4.1 Fast",
    "azure-nano":     "🔬 GPT-5.4 Nano",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Banker Copilot",
    page_icon="🏦",
    layout="wide",
)
st.title("🏦 Personal Banker Copilot")
st.caption("Pergunte sobre sua carteira de clientes em linguagem natural.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("👤 Personal Banker")
    banker_id = st.text_input("Seu ID / nome", value="banker_01", key="banker_id")

    if banker_id:
        metrics = get_banker_metrics(banker_id)
        if metrics["total_queries"] > 0:
            st.divider()
            st.subheader("📊 Suas métricas")
            col1, col2 = st.columns(2)
            col1.metric("Queries", metrics["total_queries"])
            col2.metric("Taxa de sucesso", f"{metrics['success_rate_pct']}%")
            col1.metric("Latência média", f"{metrics['avg_latency_s']}s")
            col2.metric("Falhas", metrics["failed_queries"])
            st.metric("Custo médio/query", f"${metrics['avg_cost_usd']:.5f}")
            st.caption(f"Total acumulado: ${metrics['total_cost_usd']:.4f}")
            if metrics["top_intents"]:
                st.caption("🎯 Intenções mais frequentes")
                for item in metrics["top_intents"]:
                    st.write(f"- **{item['intent']}** ({item['count']}x)")
            if metrics["recent_queries"]:
                with st.expander("🕐 Histórico recente"):
                    for q in metrics["recent_queries"]:
                        icon = "✔" if q["success"] else "✘"
                        st.write(f"{icon} {q['question'][:60]}...")
                        st.caption(f"{str(q['timestamp'])[:16]} · {q['total_latency_s']}s · {q['viz_type']}")
        else:
            st.caption("Nenhuma query registrada ainda.")

    st.divider()
    st.subheader("📋 Schema do banco")
    st.code(get_schema(), language="text")
    st.divider()
    st.caption("Powered by LangGraph + Gemini/GPT · Observabilidade: LangFuse")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🚀 Single Deploy", "⚖️ LLM Comparison"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Deploy
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    col_cfg, _ = st.columns([1, 2])
    with col_cfg:
        provider = st.selectbox(
            "Modelo",
            options=list(PROVIDER_LABELS.keys()),
            format_func=lambda x: PROVIDER_LABELS[x],
            key="tab1_provider",
        )
        os.environ["LLM_PROVIDER"] = provider  # tab1 uses env var (single thread, safe)

    EXAMPLE_QUESTIONS = [
        "Liste os 5 estados com maior número de clientes que compraram via app em maio.",
        "Quantos clientes interagiram com campanhas de WhatsApp em 2024?",
        "Quais categorias de produto tiveram o maior número de compras em média por cliente?",
        "Qual o número de reclamações não resolvidas por canal?",
        "Qual a tendência de reclamações por canal no último ano?",
    ]
    with st.expander("📋 Perguntas do enunciado", expanded=True):
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"ex_{q}"):
                st.session_state["prefill"] = q

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sql"):
                with st.expander("SQL usado"):
                    st.code(msg["sql"], language="sql")

    question = st.chat_input("Pergunte algo sobre seus clientes...") or st.session_state.pop("prefill", None)

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    final_state = run_agent(question, session_id=f"{banker_id}:{provider}")
                except Exception as e:
                    err = str(e)
                    if "RESOURCE_EXHAUSTED" in err or "429" in err:
                        model_hint = "2.5 Pro" if provider == "gemini-pro" else "2.5 Flash"
                        st.warning(
                            f"🔋 Quota diária do Gemini {model_hint} atingida. "
                            f"Tente novamente amanhã ou selecione um modelo Azure acima.",
                            icon="🔋",
                        )
                    elif "503" in err or "ServiceUnavailable" in err or "unavailable" in err.lower():
                        st.warning(
                            "⚠️ Modelo temporariamente indisponível (503). "
                            "Aguarde alguns segundos e tente novamente.",
                            icon="⚠️",
                        )
                    elif "404" in err or "not found" in err.lower() or "DeploymentNotFound" in err:
                        st.error(
                            "❌ Deployment não encontrado. Verifique se o modelo está "
                            "deployado no Foundry com o nome exato.",
                        )
                    else:
                        st.error(f"Erro inesperado: {err[:300]}")
                    st.stop()

            success = final_state.get("error") is None and final_state.get("result") is not None
            row_count = 0
            if final_state.get("result"):
                try:
                    row_count = len(pd.read_json(StringIO(final_state["result"]), orient="records"))
                except Exception:
                    pass

            tokens_text = " ".join([
                question,
                final_state.get("sql", ""),
                final_state.get("response", ""),
            ])
            log_query(
                banker_id=banker_id,
                question=question,
                intent=classify_intent(question),
                sql=final_state.get("sql", ""),
                success=success,
                retries=final_state.get("retries", 0),
                total_latency_s=total_latency(final_state["trace"]),
                viz_type=final_state.get("viz_spec", {}).get("type", ""),
                row_count=row_count,
                extra={
                    "provider": provider,
                    "estimated_cost_usd": estimate_cost_usd(tokens_text, provider),
                },
            )

            with st.expander("🧠 Raciocínio do agente", expanded=True):
                for step in enrich_trace_with_pct(final_state["trace"]):
                    st.write(step)
                if final_state.get("plan"):
                    st.divider()
                    st.caption("Plano")
                    st.write(final_state["plan"])
                if final_state.get("sql"):
                    st.divider()
                    st.caption("SQL executado")
                    st.code(final_state["sql"], language="sql")

            st.write(final_state["response"])
            render_viz(final_state, key="tab1")

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_state["response"],
                "sql": final_state.get("sql"),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LLM Comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("⚖️ Comparação de modelos — mesma pergunta, todos os LLMs")
    st.caption("Executa a query sequencialmente em cada modelo selecionado e compara latência, SQL e resposta.")

    selected = st.multiselect(
        "Modelos a comparar",
        options=list(PROVIDER_LABELS.keys()),
        default=["azure", "azure-41", "azure-nano", "azure-deepseek", "azure-grok", "gemini", "gemini-35"],
        format_func=lambda x: PROVIDER_LABELS[x],
    )

    with st.expander("📋 Perguntas do enunciado", expanded=True):
        cmp_cols = st.columns(1)
        for q in [
            "Liste os 5 estados com maior número de clientes que compraram via app em maio.",
            "Quantos clientes interagiram com campanhas de WhatsApp em 2024?",
            "Quais categorias de produto tiveram o maior número de compras em média por cliente?",
            "Qual o número de reclamações não resolvidas por canal?",
            "Qual a tendência de reclamações por canal no último ano?",
        ]:
            if st.button(q, key=f"cmp_ex_{q}"):
                st.session_state["cmp_prefill"] = q

    if "cmp_prefill" in st.session_state:
        st.session_state["cmp_question"] = st.session_state.pop("cmp_prefill")
    if "cmp_question" not in st.session_state:
        st.session_state["cmp_question"] = "Liste os 5 estados com maior número de clientes que compraram via app em maio."

    cmp_question = st.text_input(
        "Pergunta para comparar",
        key="cmp_question",
    )

    st.caption(f"⏱ Timeout por modelo: {MODEL_TIMEOUT_S}s — modelos paralelos, resultados ao vivo.")

    if st.button("🚀 Comparar modelos", disabled=not selected or not cmp_question):
        results = {}
        completed_count = 0
        progress = st.progress(0, text=f"Rodando {len(selected)} modelos em paralelo...")
        status_area = st.empty()

        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = {
                executor.submit(run_agent_for_provider, prov, cmp_question): prov
                for prov in selected
            }
            try:
                for future in as_completed(futures, timeout=MODEL_TIMEOUT_S + 5):
                    prov = futures[future]
                    completed_count += 1
                    try:
                        results[prov] = future.result(timeout=MODEL_TIMEOUT_S)
                    except Exception as e:
                        results[prov] = {
                            "success": False,
                            "error": str(e)[:200],
                            "latency": MODEL_TIMEOUT_S,
                        }
                    progress.progress(
                        completed_count / len(selected),
                        text=f"✔ {PROVIDER_LABELS[prov]} concluído — {completed_count}/{len(selected)}"
                    )
            except TimeoutError:
                # One or more models exceeded the wall-clock timeout.
                # Mark any not-yet-collected future as timed out and move on —
                # results from models that already finished are preserved.
                for future, prov in futures.items():
                    if prov not in results:
                        results[prov] = {
                            "success": False,
                            "error": f"timeout após {MODEL_TIMEOUT_S}s",
                            "latency": MODEL_TIMEOUT_S,
                        }
                        progress.progress(
                            len(results) / len(selected),
                            text=f"⏱ {PROVIDER_LABELS[prov]} excedeu {MODEL_TIMEOUT_S}s — ignorado",
                        )

        # Restore original order from multiselect (include timeouts)
        results = {p: results.get(p, {"success": False, "error": "não concluído", "latency": 0})
                   for p in selected}
        progress.progress(1.0, text="Comparação concluída ✔")
        status_area.empty()

        # ── Build rich rows for CSV + display ────────────────────────────────
        import datetime, csv
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_rows = []
        summary_rows   = []
        for prov, r in results.items():
            est_cost = 0.0
            if r["success"]:
                tokens_text = " ".join([cmp_question, r.get("sql",""), r.get("response","")])
                est_cost = estimate_cost_usd(tokens_text, prov)
            benchmark_rows.append({
                "run_id":       run_id,
                "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
                "question":     cmp_question,
                "provider":     prov,
                "model":        PROVIDER_LABELS[prov],
                "latency_s":    r["latency"] if r["success"] else None,
                "retries":      r.get("retries") if r["success"] else None,
                "cost_usd":     round(est_cost, 6) if r["success"] else None,
                "success":      r["success"],
                "sql":          r.get("sql", ""),
                "response":     r.get("response", ""),
                "error":        r.get("error", ""),
            })
            summary_rows.append({
                "Modelo":           PROVIDER_LABELS[prov],
                "Latência (s)":     r["latency"] if r["success"] else "—",
                "Retries":          r.get("retries", "—"),
                "Custo est. (USD)": f"${est_cost:.5f}" if r["success"] else "—",
                "Status":           "✔ OK" if r["success"] else "✘ Erro",
            })

        # ── Auto-save to data/llms_comparison.csv (append) ───────────────────
        csv_path = os.path.join(os.path.dirname(__file__), "data", "llms_comparison.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=benchmark_rows[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(benchmark_rows)

        # ── Summary table ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("📊 Resumo")
        summary_df = pd.DataFrame(summary_rows).astype(str)
        st.dataframe(summary_df, width="stretch", hide_index=True)

        # Export buttons (current run)
        col_dl1, col_dl2, _ = st.columns([1, 1, 3])
        with col_dl1:
            st.download_button(
                label="⬇️ Exportar esta run (CSV)",
                data=pd.DataFrame(benchmark_rows).to_csv(index=False).encode("utf-8"),
                file_name=f"llms_comparison_{run_id}.csv",
                mime="text/csv",
            )
        with col_dl2:
            if os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="📦 Baixar histórico completo",
                        data=f.read(),
                        file_name="llms_comparison.csv",
                        mime="text/csv",
                    )

        # ── Side-by-side results ──────────────────────────────────────────────
        st.divider()
        st.subheader("🔍 Respostas lado a lado")
        COLS_PER_ROW = 3
        items = list(results.items())
        for row_start in range(0, len(items), COLS_PER_ROW):
            row_items = items[row_start:row_start + COLS_PER_ROW]
            cols = st.columns(len(row_items))
            for col, (prov, r) in zip(cols, row_items):
                with col:
                    st.markdown(f"**{PROVIDER_LABELS[prov]}**")
                    if not r["success"]:
                        err = r.get("error", "")
                        if "RESOURCE_EXHAUSTED" in err or "429" in err:
                            st.warning("🔋 Quota diária atingida")
                        elif "503" in err or "ServiceUnavailable" in err:
                            st.warning("⚠️ Modelo indisponível (503)")
                        elif "404" in err or "DeploymentNotFound" in err:
                            st.error("❌ Deployment não encontrado")
                        else:
                            st.error(f"Erro: {err[:120]}")
                        continue

                    st.metric("Latência", f"{r['latency']}s")
                    st.write(r["response"])
                    render_viz(r, key=prov)

                    with st.expander("SQL"):
                        st.code(r["sql"], language="sql")

                    with st.expander("Trace"):
                        for step in enrich_trace_with_pct(r["trace"]):
                            st.write(step)

        # Restore provider to tab1 selection after comparison (env var, single thread)
        os.environ["LLM_PROVIDER"] = st.session_state.get("tab1_provider", "gemini")

    # ── Historical runs (always visible) ─────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), "data", "llms_comparison.csv")
    if os.path.exists(csv_path):
        st.divider()
        with st.expander("📂 Histórico de comparações"):
            hist_df = pd.read_csv(csv_path)
            run_options = ["Todos"] + sorted(hist_df["run_id"].unique().tolist(), reverse=True)
            selected_run = st.selectbox("Filtrar por run", run_options, key="hist_run_filter")
            if selected_run != "Todos":
                hist_df = hist_df[hist_df["run_id"] == selected_run]
            st.dataframe(
                hist_df[["timestamp","model","latency_s","retries","cost_usd","success","question","sql"]],
                width="stretch",
                hide_index=True,
            )
            st.caption(f"{len(hist_df)} registros")