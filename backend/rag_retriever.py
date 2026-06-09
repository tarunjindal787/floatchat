"""
RAG Retriever — semantic search over the ChromaDB vector store.
"""
from __future__ import annotations

from loguru import logger

from config.settings import get_settings
from ingestion.vector_indexer import semantic_search


def retrieve_context(query: str,
                     top_k: int | None = None,
                     ocean_basin: str | None = None,
                     lat_range: tuple[float, float] | None = None,
                     lon_range: tuple[float, float] | None = None) -> list[dict]:
    """
    Semantic search with optional metadata filters.

    Returns list of hit dicts: {id, document, metadata, distance}.
    """
    s = get_settings()
    k = top_k or s.rag_top_k

    filters: dict | None = None
    conditions = []

    if ocean_basin:
        conditions.append({"ocean_basin": {"$eq": ocean_basin}})
    if lat_range:
        conditions.append({"latitude": {"$gte": lat_range[0]}})
        conditions.append({"latitude": {"$lte": lat_range[1]}})
    if lon_range:
        conditions.append({"longitude": {"$gte": lon_range[0]}})
        conditions.append({"longitude": {"$lte": lon_range[1]}})

    if len(conditions) == 1:
        filters = conditions[0]
    elif len(conditions) > 1:
        filters = {"$and": conditions}

    hits = semantic_search(query, top_k=k, filters=filters)
    logger.debug(f"RAG retrieved {len(hits)} context chunks for: {query!r}")
    return hits


def format_context(hits: list[dict]) -> str:
    """Format RAG hits into a plain-text context block for LLM injection."""
    if not hits:
        return "No relevant ARGO profiles found in the vector store."
    lines = ["Relevant ARGO profile summaries (most similar first):"]
    for i, hit in enumerate(hits, 1):
        lines.append(f"\n[{i}] {hit['document']}")
        meta = hit.get("metadata", {})
        if meta.get("ocean_basin"):
            lines.append(f"    Ocean: {meta['ocean_basin']}, Date: {meta.get('juld', '?')}")
    return "\n".join(lines)
