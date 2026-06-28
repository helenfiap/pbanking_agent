# Personal Banker Copilot

> A Text-to-SQL agent built for Franq's Personal Bankers — query client behavioral data in natural language, get SQL, charts, and plain-language answers in PT-BR.

![Architecture](docs/architecture.png)

---

## The Problem

Franq operates a marketplace of 150+ financial products. Personal Bankers need to understand client behavior — purchase patterns, campaign responses, complaint trends — to match clients to the right products.

Today that means either writing SQL manually or waiting for a BI team. This copilot eliminates both bottlenecks: a banker types a question in Portuguese and gets a structured answer with a chart, the SQL that produced it, and the reasoning that got there.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd personal_banker_copilot
uv sync

# 2. Configure keys
cp .env.example .env
# Fill in GOOGLE_API_KEY and/or AZURE_OPENAI_* — see .env.example

# 3. Run
uv run streamlit run app.py
```

Requires Python 3.11+. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

---

## Architecture

The agent is implemented as a **7-node LangGraph `StateGraph`**. The challenge explicitly asked for uncertainty navigation, autonomous error correction, and reasoning transparency — conditions that map naturally to a directed graph with typed state and conditional edges.

```
START
  │
  ▼
① Schema Inspector    ← PRAGMA discovery — no LLM, always fresh
  │
  ▼
② Planner             ← Heavy LLM — reasoning plan before any SQL
  │
  ▼
③ SQL Generator       ← Heavy LLM — SQLite-specific query
  │
  ▼
④ SQL Executor        ← No LLM — runs query, catches exceptions
  │
  ├── error + retries < MAX ──▶ ⑤ Error Recovery ──┐
  │                               Cheap LLM          │ (loops back)
  │
  ├── max retries exceeded ──▶ ⑦ Response Formatter
  │
  └── success ──▶ ⑥ Visualization Agent ──▶ ⑦ Response Formatter ──▶ END
                    Hybrid: rules + cheap LLM
```

### Shared State

```python
class AgentState(TypedDict):
    question: str       # original user input
    schema:   str       # live PRAGMA output
    plan:     str       # planner reasoning
    sql:      str       # current SQL (mutated by recovery)
    result:   str       # JSON string of query rows
    error:    str       # last SQLite exception
    retries:  int       # retry counter
    viz_spec: dict      # {type, x, y, orientation, title}
    response: str       # final PT-BR natural language answer
    trace:    list[str] # per-node latency steps
```

---

## Engineering Decisions

### Why LangGraph?

The challenge asked for an agent that "navigates uncertainty, seeks its own answers, detects errors and corrects itself." Those are state transitions, not a linear chain. LangGraph models them as first-class graph edges — the retry loop between executor and error recovery is a real conditional edge, not a `for` loop inside a monolith.

### Why a Planner node before SQL generation?

Complex multi-join questions fail at SQL generation without prior reasoning. The planner forces the heavy LLM to identify which tables are needed and why before writing any SQL. In testing, this reduced hallucinated column names and missing JOINs significantly.

### Why hybrid visualization routing?

Calling an LLM to decide "should this be a table or a chart?" costs tokens on every query. The visualization agent uses three layers ordered by cost:

1. **Deterministic pre-filters** (no LLM): single scalar → `metric`, >15 rows → `table`, single row → `table`
2. **Keyword intent classifier** (no LLM): "top/maior/menor" → ranking → horizontal bar; "tendência/mês a mês" → line
3. **Cheap LLM** (only for ambiguous 2-15 row results): receives intent hint + column types + sample data

~80% of queries never reach the LLM call. This matters at scale.

### Why provider abstraction?

LLM selection is an engineering decision, not a product decision — it has latency, cost, and reliability constraints that change over time. Isolating provider choice behind a common interface means the seven nodes never change when the underlying model changes. From the factory:

```python
def get_llm(tier: str = "heavy") -> BaseChatModel:
    # resolves thread-local → env var → default
    # returns ChatOpenAI or ChatGoogleGenerativeAI
    # caller never knows which
```

This also enabled the benchmark tab — see below.

### Why a benchmark tab?

Initially the challenge called for a single-model deploy. During development it became clear that choosing the right LLM is itself part of the engineering problem. Rather than doing that analysis offline in a notebook, I exposed it as a tool inside the app:

- All models run in parallel via `ThreadPoolExecutor`
- Results appear live as each model finishes
- Every run appends to `data/llms_comparison.csv` for historical comparison

The benchmark is a **development and operations tool**, not a user-facing feature. It answers the question "which model should we deploy to production?" with real data from the actual workload.

---

## LLM Cost Analysis

Results from live benchmark runs against a campaign effectiveness question:

| Model | Provider | Latency | Cost/query | Notes |
|---|---|---|---|---|
| GPT-5.4 Nano | Azure Foundry | 8.7s | **$0.00011** | Best value, concise answers |
| Kimi K2.5 | Azure Foundry | ~48s | $0.00036 | Uniquely returns revenue in BRL |
| GPT-4.1 | Azure Foundry | 6.4s | $0.00054 | **Sweet spot** — half the cost of 4o |
| Grok 4.1 Fast | Azure Foundry | 8.4s | $0.00054 | Strong % metrics framing |
| GPT-4o | Azure Foundry | 7.3s | $0.00108 | Most narrative, highest cost |
| Gemini 2.5 Flash | Google AI Studio | ~8s | $0.00030 | Free tier: 40 req/day with 2-key rotation |
| Gemini 2.0 Flash Lite | Google AI Studio | ~5s | $0.00010 | Free tier: 1,500 req/day — best for dev |

**Recommendation for production:** GPT-4.1 as default (cost/quality balance). Gemini 2.0 Flash Lite for development and high-volume internal tooling.

Each model interprets the same SQL result differently — GPT-4o emphasizes narrative, Kimi returns absolute revenue, Grok uses percentage framing. That's useful signal when deciding which model to surface to end users.

---

## Sample Queries

All five questions from the challenge brief, tested end-to-end:

```sql
-- 1. Top states by client count
"Quais os 5 estados com maior número de clientes?"

-- 2. Campaign effectiveness
"Quais as campanhas de marketing que foram mais efetivas em 2024?"

-- 3. Monthly complaint trend
"Qual a tendência de reclamações por canal no último ano?"

-- 4. Unresolved complaints by channel
"Qual o número de reclamações não resolvidas por canal?"

-- 5. Product categories by average purchases per client
"Quais categorias de produto tiveram o maior número de compras em média por cliente?"
```

The agent returns SQL, a Plotly chart (bar, line, pie, table, or metric depending on result shape), and a 2-3 sentence PT-BR explanation.

---

## Observability

LangFuse is wired into `graph.invoke()` via optional callbacks. Every run produces:

- One trace per query with per-node spans
- Token counts and latency per node
- Session IDs tagged `{banker_id}:{provider}` (Tab 1) or `benchmark:{provider}` (Tab 2)

If LangFuse keys are not set, the agent runs normally — the integration degrades gracefully to `None`.

Per-banker usage is also logged locally to `data/logs/<banker_id>.jsonl` with cost estimates, intent classification, and retry counts. This feeds the sidebar metrics panel and is designed for a Parquet → DuckDB upgrade path when volume grows.

---

## Scaling to Production

This demo runs on SQLite + Google AI Studio free tier. Here is what changes at Franq's scale:

| Layer | Demo | Production |
|---|---|---|
| Database | SQLite (local file) | Cloud SQL (PostgreSQL) or BigQuery |
| LLM | Google AI Studio free tier | Vertex AI Gemini or Azure OpenAI with SLAs |
| Auth | None | IAM + per-banker JWT |
| Logging | JSONL → local disk | Parquet → GCS → BigQuery |
| Deployment | Streamlit Community Cloud | Cloud Run (containerized) |
| Secrets | `.env` file | Secret Manager |
| Schema discovery | PRAGMA on startup | Cached metadata catalog with refresh |

The agent code itself does not change between environments. Provider selection, database connection, and logging destination are all configuration — not logic.

### What a production deploy looks like

```
Personal Banker (browser)
        │
        ▼
   Cloud Run (Streamlit)
        │
        ├──▶ Vertex AI Gemini (heavy nodes)
        ├──▶ Vertex AI Gemini Flash (cheap nodes)
        │
        ├──▶ Cloud SQL (query execution)
        │
        ├──▶ LangFuse (observability)
        │
        └──▶ GCS / BigQuery (usage logs → banker analytics)
```

---

## Known Limitations

**Multi-turn memory not implemented.** Each question starts fresh — the agent has no context from previous questions in the same session. Follow-up questions like "show me the same data for São Paulo only" require the full question to be restated.

**Schema-dependent quality.** SQL quality degrades if column names are ambiguous or if the question requires domain knowledge not present in the schema (e.g., "high-value clients" has no definition in the database).

**Free-tier Gemini quota.** Google AI Studio limits free usage to 40 requests/day (with two-key rotation). The Azure Foundry models have no such limit.

**Kimi K2.5 latency.** Azure Foundry adds routing overhead for third-party models. Kimi consistently runs 45-55 seconds — above the 45s benchmark timeout. Not a code issue; infrastructure constraint.

**No SQL injection guardrails.** The generated SQL is run directly against the database. In production, a read-only database user and query allowlist would be required.

---

## Roadmap

**Deploy 1 — Multi-turn memory**
Add `ConversationBufferMemory` or `RunnableWithMessageHistory` so bankers can refine queries conversationally. "Show me the same for SP" should work.

**Deploy 2 — Production hardening**
Read-only DB user, query timeout, result row cap, PII redaction in logs, Cloud Run container.

**Deploy 3 — Banker analytics**
Aggregate all `data/logs/*.jsonl` → Parquet → DuckDB. Surface patterns: which questions are asked most, which fail most, which cost most. Feed back into prompt improvements.

---

## Project Structure

```
personal_banker_copilot/
├── agent/
│   ├── graph.py                   # LangGraph StateGraph + routing
│   ├── state.py                   # AgentState TypedDict
│   └── nodes/
│       ├── schema_inspector.py    # Node 1 — PRAGMA, no LLM
│       ├── planner.py             # Node 2 — heavy LLM
│       ├── sql_generator.py       # Node 3 — heavy LLM
│       ├── sql_executor.py        # Node 4 — no LLM
│       ├── error_recovery.py      # Node 5 — cheap LLM
│       ├── visualization_agent.py # Node 6 — hybrid routing
│       └── response_formatter.py  # Node 7 — cheap LLM
├── utils/
│   ├── llm.py                     # Provider-agnostic factory (7 providers)
│   ├── llm_output.py              # extract_text() — handles LangChain content blocks
│   ├── database.py                # Dynamic schema discovery
│   └── logger.py                  # JSONL per-banker logging + cost estimates
├── data/
│   └── franq.db                   # Challenge SQLite database
├── docs/
│   └── architecture.png           # LangGraph flow diagram
├── devlog/
│   ├── p1.md                      # Day 1-2 build log
│   └── p2.md                      # Day 2-3 fixes and benchmark log
├── app.py                         # Streamlit UI (Single Deploy + LLM Comparison tabs)
├── pyproject.toml                 # uv-compatible dependencies
├── .env.example                   # Key placeholders — copy to .env
└── .gitignore
```

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Agent orchestration | LangGraph | Graph models retries, uncertainty, and trace naturally |
| LLM (heavy) | Gemini 2.5 Flash / GPT-4.1 | Strong SQL reasoning, configurable |
| LLM (cheap) | Gemini 2.0 Flash Lite / GPT-5.4 Nano | Cost-efficient for formatting and viz |
| UI | Streamlit | Fast iteration, free cloud deploy |
| Database | SQLite | Zero-infra, matches challenge data |
| Observability | LangFuse | Per-node spans, token counts, latency |
| Package manager | uv | 10x faster than pip |
| Parallelism | ThreadPoolExecutor | Benchmark runs all models simultaneously |
