"""
Vector Indexer — embeds Argo profile summaries and stores in ChromaDB.
Supports sentence-transformers (local) and OpenAI embeddings.
"""
from __future__ import annotations

import json
import chromadb
from chromadb.utils import embedding_functions
from loguru import logger

from config.settings import get_settings


def get_chroma_client():
    s = get_settings()
    return chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)


def get_collection(client=None):
    s = get_settings()
    if client is None:
        client = get_chroma_client()

    if s.embedding_provider == "openai":
        ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=s.openai_api_key,
            model_name="text-embedding-3-small",
        )
    else:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=s.embedding_model
        )

    return client.get_or_create_collection(
        name=s.chroma_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def build_profile_summary(row: dict) -> str:
    """
    Generate a plain-English summary for a profile row.
    row keys: float_id, cycle_number, juld, latitude, longitude,
              ocean_basin, mean_temp, mean_salinity, max_pressure
    """
    lat  = row.get("latitude")
    lon  = row.get("longitude")
    lat_str = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}" if lat is not None else "unknown"
    lon_str = f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}" if lon is not None else "unknown"

    juld = row.get("juld", "unknown date")
    basin = row.get("ocean_basin", "Unknown Ocean")
    mean_t = row.get("mean_temp")
    mean_s = row.get("mean_salinity")
    max_p  = row.get("max_pressure")

    temp_str  = f"mean temperature {mean_t:.2f}°C" if mean_t is not None else "temperature unavailable"
    psal_str  = f"mean salinity {mean_s:.2f} PSU" if mean_s is not None else "salinity unavailable"
    depth_str = f"max depth {max_p:.0f} dbar" if max_p is not None else "depth unknown"

    return (
        f"Argo float {row.get('float_id')} cycle {row.get('cycle_number')}: "
        f"{basin}, {lat_str} {lon_str} on {juld}. "
        f"{temp_str.capitalize()}, {psal_str}, {depth_str}."
    )


def index_profiles(profiles: list[dict], batch_size: int = 256) -> int:
    """
    Embed and store profile summaries in ChromaDB.
    profiles: list of dicts with profile metadata + stats.
    Returns number of documents indexed.
    """
    collection = get_collection()
    total = 0

    for i in range(0, len(profiles), batch_size):
        batch = profiles[i : i + batch_size]
        docs, ids, metas = [], [], []

        for p in batch:
            doc_id = f"{p['float_id']}_{p['cycle_number']}_{p.get('direction', 'A')}"
            summary = build_profile_summary(p)
            meta = {
                "float_id":     str(p.get("float_id", "")),
                "cycle_number": int(p.get("cycle_number", 0)),
                "latitude":     float(p["latitude"]) if p.get("latitude") is not None else 0.0,
                "longitude":    float(p["longitude"]) if p.get("longitude") is not None else 0.0,
                "ocean_basin":  str(p.get("ocean_basin", "")),
                "juld":         str(p.get("juld", "")),
                "profile_id":   int(p.get("profile_id", 0)),
            }
            docs.append(summary)
            ids.append(doc_id)
            metas.append(meta)

        collection.upsert(documents=docs, ids=ids, metadatas=metas)
        total += len(batch)
        logger.debug(f"Indexed {total}/{len(profiles)} profiles")

    logger.success(f"Vector index updated: {total} profiles")
    return total


def semantic_search(query: str, top_k: int = 8,
                    filters: dict | None = None) -> list[dict]:
    """
    Search the vector store for profiles matching `query`.
    Returns list of {id, document, metadata, distance}.
    """
    collection = get_collection()
    where = filters if filters else None

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    if results and results["ids"]:
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "id":       doc_id,
                "document": doc,
                "metadata": meta,
                "distance": dist,
            })
    return hits


def delete_collection() -> None:
    """Drop the entire vector collection (for reindexing)."""
    s = get_settings()
    client = get_chroma_client()
    client.delete_collection(s.chroma_collection_name)
    logger.warning(f"Deleted ChromaDB collection: {s.chroma_collection_name}")
