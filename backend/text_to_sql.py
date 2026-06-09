"""
Text-to-SQL module — translates natural language to safe PostgreSQL SELECT queries.
"""
from __future__ import annotations

import re

import sqlglot
from loguru import logger

from backend.models.llm_client import LLMClient, Message

# ── Schema description sent to the LLM ───────────────────────────────────────
SCHEMA_DESCRIPTION = """
PostgreSQL database for ARGO oceanographic float data.

Tables:
1. floats(float_id PK, platform_number, dac, ocean_basin, first_seen DATE, last_seen DATE)
2. profiles(profile_id PK, float_id FK, cycle_number INT, juld TIMESTAMPTZ,
            latitude DOUBLE, longitude DOUBLE, position_qc CHAR, direction CHAR, data_mode CHAR)
3. measurements(meas_id PK, profile_id FK, pressure REAL, temperature REAL, salinity REAL,
                temp_adjusted REAL, psal_adjusted REAL, temp_qc CHAR, psal_qc CHAR)
4. bgc_data(bgc_id PK, profile_id FK, pressure REAL, doxy REAL, chla REAL,
            nitrate REAL, ph_in_situ REAL, bbp700 REAL)

Views:
- float_summary  → aggregated stats per float
- profile_stats  → aggregated stats per profile

Useful SQL patterns:
- Filter by region: WHERE latitude BETWEEN <min> AND <max> AND longitude BETWEEN <min> AND <max>
- Filter by date: WHERE juld BETWEEN '<start>'::timestamptz AND '<end>'::timestamptz
- Filter by ocean: WHERE f.ocean_basin ILIKE '%Indian%'
- Nearest float: ORDER BY ST_Distance(ST_MakePoint(longitude, latitude), ST_MakePoint(<lon>, <lat>)) LIMIT <n>
- Indian Ocean rough bbox: latitude BETWEEN -60 AND 30, longitude BETWEEN 20 AND 120
- Arabian Sea bbox: latitude BETWEEN 5 AND 25, longitude BETWEEN 50 AND 78
- Bay of Bengal bbox: latitude BETWEEN 5 AND 22, longitude BETWEEN 80 AND 100
"""

SYSTEM_PROMPT = f"""You are an expert oceanographer and PostgreSQL developer.
Given the schema below and a user question, write a single valid PostgreSQL SELECT query.
Rules:
- Output ONLY the SQL query, no explanation, no markdown fences.
- Only use SELECT statements. Never use INSERT, UPDATE, DELETE, DROP, CREATE, or EXECUTE.
- Use table aliases. Always JOIN floats f when you need ocean_basin.
- Limit results to 2000 rows unless user asks for more.
- For temperature use the measurements table, join via profile_id.
- Use ILIKE for text comparisons.

Schema:
{SCHEMA_DESCRIPTION}
"""


def translate_to_sql(user_question: str) -> str:
    """
    Translate a natural language question to a SQL query.
    Returns the validated SQL string.
    Raises ValueError if the LLM output cannot be safely parsed.
    """
    client = LLMClient()
    messages = [
        Message("system", SYSTEM_PROMPT),
        Message("user", user_question),
    ]

    raw = client.complete_sync(messages, temperature=0.0, max_tokens=512)
    logger.debug(f"LLM SQL output: {raw!r}")

    sql = _extract_sql(raw)
    _validate_sql(sql)
    return sql


def _extract_sql(text: str) -> str:
    """Strip markdown code fences and extra whitespace."""
    text = text.strip()
    # Remove ```sql ... ``` or ``` ... ```
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip().rstrip(";")


def _validate_sql(sql: str) -> None:
    """
    Parse SQL with sqlglot and ensure it is a SELECT-only statement.
    Raises ValueError on any non-SELECT or dangerous pattern.
    """
    # Block dangerous keywords
    dangerous = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXECUTE|COPY|GRANT|REVOKE)\b",
        re.IGNORECASE,
    )
    if dangerous.search(sql):
        raise ValueError("Unsafe SQL statement detected and blocked.")

    # Parse with sqlglot for structural validation
    try:
        parsed = sqlglot.parse_one(sql, dialect="postgres")
    except Exception as exc:
        raise ValueError(f"SQL parse error: {exc}") from exc

    # Must be a SELECT node
    if parsed.__class__.__name__ != "Select":
        raise ValueError(f"Only SELECT queries are allowed. Got: {parsed.__class__.__name__}")

    logger.debug("SQL validation passed.")
