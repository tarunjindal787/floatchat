"""
Tests for the Text-to-SQL module — extraction, validation, and safety.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from backend.text_to_sql import _extract_sql, _validate_sql


# ── SQL extraction tests ──────────────────────────────────────────────────────

def test_extract_strips_markdown_fences():
    raw = "```sql\nSELECT * FROM profiles LIMIT 10\n```"
    sql = _extract_sql(raw)
    assert sql == "SELECT * FROM profiles LIMIT 10"


def test_extract_strips_plain_fences():
    raw = "```\nSELECT float_id FROM floats\n```"
    assert _extract_sql(raw) == "SELECT float_id FROM floats"


def test_extract_strips_trailing_semicolon():
    raw = "SELECT * FROM floats;"
    assert _extract_sql(raw) == "SELECT * FROM floats"


def test_extract_plain_sql():
    raw = "SELECT * FROM profiles WHERE latitude > 0"
    assert _extract_sql(raw) == raw


# ── SQL validation tests ──────────────────────────────────────────────────────

def test_validate_valid_select():
    """Valid SELECT should not raise."""
    _validate_sql("SELECT float_id, latitude FROM profiles LIMIT 100")


def test_validate_rejects_drop():
    with pytest.raises(ValueError, match="Unsafe SQL"):
        _validate_sql("DROP TABLE profiles")


def test_validate_rejects_insert():
    with pytest.raises(ValueError, match="Unsafe SQL"):
        _validate_sql("INSERT INTO floats (float_id) VALUES ('test')")


def test_validate_rejects_delete():
    with pytest.raises(ValueError, match="Unsafe SQL"):
        _validate_sql("DELETE FROM measurements")


def test_validate_rejects_update():
    with pytest.raises(ValueError, match="Unsafe SQL"):
        _validate_sql("UPDATE floats SET ocean_basin='test'")


def test_validate_rejects_execute():
    with pytest.raises(ValueError, match="Unsafe SQL"):
        _validate_sql("EXECUTE some_procedure()")


def test_validate_complex_select():
    """Complex JOIN query should pass validation."""
    sql = """
        SELECT p.float_id, p.juld, AVG(m.temperature) as mean_temp
        FROM profiles p
        JOIN measurements m ON p.profile_id = m.profile_id
        JOIN floats f ON p.float_id = f.float_id
        WHERE f.ocean_basin ILIKE '%Indian%'
          AND p.juld BETWEEN '2023-01-01' AND '2023-12-31'
        GROUP BY p.float_id, p.juld
        ORDER BY p.juld
        LIMIT 500
    """
    _validate_sql(sql)  # should not raise


def test_validate_rejects_invalid_syntax():
    """Malformed SQL should raise ValueError."""
    with pytest.raises(ValueError):
        _validate_sql("SELEKT * FORM profiles")


# ── translate_to_sql integration (mocked) ────────────────────────────────────

@patch("backend.text_to_sql.LLMClient")
def test_translate_to_sql_calls_llm(MockLLM):
    from backend.text_to_sql import translate_to_sql

    mock_instance = MagicMock()
    mock_instance.complete_sync.return_value = (
        "SELECT float_id, latitude, longitude FROM profiles "
        "WHERE latitude BETWEEN -5 AND 5 LIMIT 100"
    )
    MockLLM.return_value = mock_instance

    sql = translate_to_sql("Show floats near the equator")
    assert "SELECT" in sql.upper()
    assert "profiles" in sql.lower()
