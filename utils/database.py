import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "franq.db"


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def run_query(sql: str) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn)


def get_schema() -> str:
    """
    Dynamically reads schema including distinct values for low-cardinality
    TEXT/BOOLEAN columns. This prevents the LLM from generating wrong string
    literals — SQLite is case-sensitive ('App' != 'app', 'WhatsApp' != 'whatsapp').
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

            col_lines = []
            for col in columns:
                col_name = col[1]
                col_type = col[2].upper()
                line = f"  - {col_name} ({col_type})"

                if col_type in ("TEXT", "BOOLEAN"):
                    cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {table}")
                    n_distinct = cursor.fetchone()[0]
                    if n_distinct <= 20:
                        cursor.execute(
                            f"SELECT DISTINCT {col_name} FROM {table} "
                            f"WHERE {col_name} IS NOT NULL ORDER BY {col_name} LIMIT 20"
                        )
                        vals = [str(r[0]) for r in cursor.fetchall()]
                        line += f" — valores exatos: {vals}"
                    elif any(p in col_name.lower() for p in ("data", "date")):
                        cursor.execute(f"SELECT MIN({col_name}), MAX({col_name}) FROM {table}")
                        mn, mx = cursor.fetchone()
                        line += f" — range: '{mn}' a '{mx}'"

                col_lines.append(line)

            schema_parts.append(f"Table: {table} ({count} rows)\n" + "\n".join(col_lines))

        schema_parts.append(
            "ATENCAO - SQLite e case-sensitive para strings. "
            "Use EXATAMENTE os valores listados acima "
            "('App' nao 'app', 'WhatsApp' nao 'whatsapp', 'E-mail' nao 'email'). "
            "Para filtrar por ano/mes use strftime('%Y', coluna) = '2024'."
        )

        return "\n\n".join(schema_parts)
