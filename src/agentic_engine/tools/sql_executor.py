import sqlite3
import pandas as pd
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class SQLQueryTool:
    """
    Sandboxed In-Memory SQLite execution tool for agentic tabular data reasoning.
    Enables autonomous agents to inspect schemas, generate SELECT queries, and summarize results.
    """
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._seed_sample_tables()

    def _seed_sample_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS energy_production (
                facility_id TEXT PRIMARY KEY,
                facility_name TEXT,
                technology TEXT,
                capacity_mw REAL,
                location_state TEXT,
                annual_yield_mwh REAL
            );
        """)
        sample_rows = [
            ("FAC_01", "Bavaria Solar Park East", "Solar PV", 45.2, "Bavaria", 51200.0),
            ("FAC_02", "Nordsee Wind Farm Alpha", "Offshore Wind", 120.0, "Lower Saxony", 380000.0),
            ("FAC_03", "Regensburg Bio-Energy Plant", "Biomass", 18.5, "Bavaria", 112000.0),
            ("FAC_04", "Black Forest Hydro Station", "Hydroelectric", 30.0, "Baden-Württemberg", 95000.0),
            ("FAC_05", "Munich Rooftop Solar Cluster", "Solar PV", 12.8, "Bavaria", 14500.0)
        ]
        cursor.executemany("INSERT OR REPLACE INTO energy_production VALUES (?,?,?,?,?,?)", sample_rows)
        self.conn.commit()

    def get_schema(self) -> Dict[str, List[str]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        schema = {}
        for t in tables:
            cursor.execute(f"PRAGMA table_info({t});")
            schema[t] = [f"{col[1]} ({col[2]})" for col in cursor.fetchall()]
        return schema

    def execute_query(self, query: str) -> Dict[str, Any]:
        # Enforce read-only constraint for security
        clean_q = query.strip().upper()
        if not clean_q.startswith("SELECT") and not clean_q.startswith("WITH"):
            return {
                "success": False,
                "error": "Security Guardrail Violation: Only read-only SELECT queries are permitted."
            }
        try:
            df = pd.read_sql_query(query, self.conn)
            return {
                "success": True,
                "row_count": len(df),
                "columns": list(df.columns),
                "data": df.head(50).to_dict(orient="records")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
