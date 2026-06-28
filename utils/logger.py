"""
Per-banker query logger — JSONL format.

Why JSONL instead of SQLite:
- Schema-free: add new fields (token_count, model_used, etc.) without migrations
- Natively reads into pandas, Spark, and MLflow with pd.read_json(lines=True)
- Easy to stream to GCS/S3 as a data lake layer
- Each line is a valid JSON object → compatible with LangFuse export format
- Future: load into a feature store or MLflow experiment for prediction models

One file per banker: data/logs/<banker_id>.jsonl
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# Blended price per 1M tokens (input + output average), USD
# Sources: official pricing pages — free tier = $0 for quota purposes but shown as list price
PRICE_PER_1M = {
    "gemini":      0.30,   # gemini-3.5-flash:      $0.15 in / $0.60 out
    "gemini-lite": 0.10,   # gemini-3.1-flash-lite: $0.075 in / $0.30 out
    "gemini-pro": 5.00,   # gemini-2.5-pro:    $1.25 in / $10 out
    "azure":      6.25,   # gpt-4o:            $2.50 in / $10 out
    "azure-41":   4.00,   # gpt-4.1:           $2.00 in / $8 out
    "azure-kimi": 0.90,   # kimi-k2.5:         ~$0.15 in / $2.50 out (approx)
    "azure-grok": 3.00,   # grok-4-1-fast:     approx
    "azure-nano": 0.50,   # gpt-5.4-nano:      approx cheapest GPT-5 tier
}


def estimate_cost_usd(text: str, provider: str) -> float:
    """
    Rough cost estimate: characters / 4 ≈ tokens, × price per 1M.
    Not exact (no tokenizer), but consistent enough for relative comparison.
    """
    approx_tokens = len(text) / 4
    price = PRICE_PER_1M.get(provider, 0.0)
    return round(approx_tokens / 1_000_000 * price, 6)

LOGS_DIR = Path(__file__).parent.parent / "data" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _log_path(banker_id: str) -> Path:
    safe_id = banker_id.replace(" ", "_").lower()
    return LOGS_DIR / f"{safe_id}.jsonl"


def log_query(
    banker_id: str,
    question: str,
    intent: str,
    sql: str,
    success: bool,
    retries: int,
    total_latency_s: float,
    viz_type: str,
    row_count: int,
    extra: dict | None = None,  # open field: add token_count, model, cost, etc.
) -> None:
    """
    Append one query run as a JSON line.
    New fields can be added to `extra` at any time without breaking existing logs.
    """
    record = {
        "timestamp":       datetime.utcnow().isoformat(),
        "banker_id":       banker_id,
        "question":        question,
        "intent":          intent,
        "sql":             sql,
        "success":         success,
        "retries":         retries,
        "total_latency_s": total_latency_s,
        "viz_type":        viz_type,
        "row_count":       row_count,
        **(extra or {}),
    }
    with _log_path(banker_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_banker_df(banker_id: str) -> pd.DataFrame:
    """Load all logs for a banker into a DataFrame. Returns empty DF if no logs yet."""
    path = _log_path(banker_id)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def get_banker_metrics(banker_id: str) -> dict:
    """
    Aggregate CX metrics for the sidebar panel.
    Pure pandas — no SQL needed, directly ML-pipeline-compatible.
    """
    df = _load_banker_df(banker_id)
    if df.empty:
        return {
            "total_queries": 0,
            "success_rate_pct": 0.0,
            "avg_latency_s": 0.0,
            "avg_retries": 0.0,
            "failed_queries": 0,
            "top_intents": [],
            "recent_queries": [],
        }

    top_intents = (
        df.groupby("intent")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(5)
        .to_dict("records")
    )

    recent = (
        df.sort_values("timestamp", ascending=False)
        .head(5)[["timestamp", "question", "success", "total_latency_s", "viz_type"]]
        .to_dict("records")
    )

    avg_cost = round(df["estimated_cost_usd"].mean(), 6) if "estimated_cost_usd" in df.columns else 0.0
    total_cost = round(df["estimated_cost_usd"].sum(), 4) if "estimated_cost_usd" in df.columns else 0.0

    return {
        "total_queries":      len(df),
        "success_rate_pct":   round(df["success"].mean() * 100, 1),
        "avg_latency_s":      round(df["total_latency_s"].mean(), 2),
        "avg_retries":        round(df["retries"].mean(), 2),
        "failed_queries":     int((~df["success"]).sum()),
        "avg_cost_usd":       avg_cost,
        "total_cost_usd":     total_cost,
        "top_intents":        top_intents,
        "recent_queries":     recent,
    }


def get_all_bankers_df() -> pd.DataFrame:
    """
    Load and merge all banker logs into one DataFrame.
    Ready to plug into MLflow, scikit-learn, or a feature store.

    Usage:
        df = get_all_bankers_df()
        mlflow.log_artifact(df)          # log to MLflow experiment
        df.to_parquet("logs/all.parquet") # convert to Parquet for Spark
    """
    files = list(LOGS_DIR.glob("*.jsonl"))
    if not files:
        return pd.DataFrame()
    return pd.concat(
        [pd.read_json(f, lines=True) for f in files],
        ignore_index=True
    )
