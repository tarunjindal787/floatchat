"""
Tests for the RAG retriever and query router.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.query_router import route_query, QueryIntent


# ── Query Router tests ────────────────────────────────────────────────────────

def test_route_sql_temperature():
    r = route_query("Show me temperature profiles in the Arabian Sea in March 2023")
    assert r.intent == QueryIntent.SQL


def test_route_trajectory():
    r = route_query("Show the trajectory of float 6902742")
    assert r.intent == QueryIntent.TRAJECTORY


def test_route_nearest():
    r = route_query("What are the nearest floats to 12.5, 68.3?")
    assert r.intent == QueryIntent.NEAREST
    assert r.entities.get("latitude") == pytest.approx(12.5, abs=0.1)


def test_route_export():
    r = route_query("Download float 6902742 data as NetCDF")
    assert r.intent == QueryIntent.EXPORT


def test_route_rag_explanation():
    r = route_query("What is the Argo program?")
    assert r.intent in (QueryIntent.RAG, QueryIntent.GENERAL)


def test_route_extracts_float_id():
    r = route_query("Show profile for float 2902733 cycle 10")
    assert "2902733" in r.entities.get("float_ids", [])


def test_route_general_greeting():
    r = route_query("Hello!")
    assert r.intent == QueryIntent.GENERAL


# ── RAG Retriever tests ───────────────────────────────────────────────────────

def test_format_context_empty():
    from backend.rag_retriever import format_context
    result = format_context([])
    assert "No relevant" in result


def test_format_context_with_hits():
    from backend.rag_retriever import format_context
    hits = [
        {
            "document": "Float 6902742 cycle 10: Arabian Sea, 15°N 65°E.",
            "metadata": {"float_id": "6902742", "ocean_basin": "Indian Ocean", "juld": "2023-03-15"},
            "distance": 0.12,
        }
    ]
    ctx = format_context(hits)
    assert "6902742" in ctx
    assert "Indian Ocean" in ctx


@patch("backend.rag_retriever.semantic_search")
def test_retrieve_context_calls_search(mock_search):
    from backend.rag_retriever import retrieve_context
    mock_search.return_value = []
    result = retrieve_context("salinity profiles Arabian Sea", top_k=5)
    mock_search.assert_called_once()
    assert result == []
