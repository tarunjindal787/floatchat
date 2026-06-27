"""
Query Router — classifies user intent and routes to SQL or RAG pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryIntent(str, Enum):
    SQL       = "sql"        # structured data query → Text-to-SQL
    RAG       = "rag"        # knowledge/explanation → vector search
    TRAJECTORY= "trajectory" # float path visualization
    NEAREST   = "nearest"    # nearest float to coordinates
    EXPORT    = "export"     # data download
    GENERAL   = "general"    # general chat / greeting


@dataclass
class RouterResult:
    intent: QueryIntent
    confidence: float
    entities: dict = field(default_factory=dict)


# ── Keyword heuristics ────────────────────────────────────────────────────────
_SQL_KEYWORDS = re.compile(
    r"\b(show|list|find|how many|count|average|mean|compare|between|"
    r"temperature|salinity|pressure|profile|float|depth|cycle|"
    r"march|april|may|june|july|august|2022|2023|2024|2025|"
    r"arabian sea|bay of bengal|indian ocean|pacific|atlantic|"
    r"equator|latitude|longitude)\b",
    re.IGNORECASE,
)

_RAG_KEYWORDS = re.compile(
    r"\b(what is|explain|describe|tell me about|how does|why|"
    r"difference between|overview|summary|argo program|bgc|ctd|"
    r"doxy|chlorophyll|nitrate|backscatter|quality flag|qc)\b",
    re.IGNORECASE,
)

_TRAJECTORY_KEYWORDS = re.compile(
    r"\b(trajectory|path|track|drift|movement|route|where has float)\b",
    re.IGNORECASE,
)

_NEAREST_KEYWORDS = re.compile(
    r"\b(nearest|closest|near|close to|around|within)\b",
    re.IGNORECASE,
)

_EXPORT_KEYWORDS = re.compile(
    r"\b(export|download|save|netcdf|csv|parquet|ascii|file)\b",
    re.IGNORECASE,
)

# Coordinate extractor
_LAT_LON = re.compile(
    r"(?P<lat>-?\d+\.?\d*)\s*[°N|°S|°]?\s*[,/\s]+\s*(?P<lon>-?\d+\.?\d*)\s*[°E|°W]?",
    re.IGNORECASE,
)

_FLOAT_ID = re.compile(r"\b(\d{7})\b")


def route_query(user_input: str) -> RouterResult:
    """
    Route a user query to the appropriate handler.
    Uses keyword matching; can be upgraded to an LLM classifier.
    """
    text = user_input.strip()
    entities: dict = {}

    # Extract float IDs
    fids = _FLOAT_ID.findall(text)
    if fids:
        entities["float_ids"] = fids

    # Extract coordinates
    coord_match = _LAT_LON.search(text)
    if coord_match:
        entities["latitude"]  = float(coord_match.group("lat"))
        entities["longitude"] = float(coord_match.group("lon"))

    # Score intents
    scores = {
        QueryIntent.EXPORT:     len(_EXPORT_KEYWORDS.findall(text)) * 3,
        QueryIntent.NEAREST:    len(_NEAREST_KEYWORDS.findall(text)) * 3,
        QueryIntent.TRAJECTORY: len(_TRAJECTORY_KEYWORDS.findall(text)) * 3,
        QueryIntent.RAG:        len(_RAG_KEYWORDS.findall(text)) * 2,
        QueryIntent.SQL:        len(_SQL_KEYWORDS.findall(text)),
    }

    # Nearest requires coordinates or "nearest float"
    if scores[QueryIntent.NEAREST] > 0 and "latitude" not in entities:
        scores[QueryIntent.NEAREST] = 0

    best_intent = max(scores, key=lambda k: scores[k])
    best_score  = scores[best_intent]

    if best_score == 0:
        best_intent = QueryIntent.GENERAL
        confidence  = 0.5
    else:
        total = sum(scores.values()) or 1
        confidence = best_score / total

    return RouterResult(intent=best_intent, confidence=confidence, entities=entities)
