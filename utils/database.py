import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "franq.db"


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def run_query(sql: str) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn)


def get_schema() -> str:
    """
    Dynamically reads the database schema — no hardcoded table names.
    Returns a human-readable string describing tables, columns, types, and row counts.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall() if row[0] != "sqlite_sequence"]

        schema_parts = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]

            col_defs = ", ".join([f"{col[1]} {col[2]}" for col in columns])
            schema_parts.append(f"Table: {table} ({count} rows)\nColumns: {col_defs}")

        return "\n\n".join(schema_parts)
