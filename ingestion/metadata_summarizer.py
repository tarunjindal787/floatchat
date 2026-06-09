"""
Metadata summarizer — generates LLM-based plain-English summaries for Argo profiles.
Used to enrich vector store documents with more descriptive text.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from backend.models.llm_client import LLMClient, Message


SUMMARY_SYSTEM = """You are a marine scientist summarizing Argo float profile data.
Write a single concise sentence (max 60 words) describing the oceanographic context
of the given profile. Include: float ID, location, date, ocean basin, and key
physical characteristics (temperature range, salinity, depth). Be factual and precise.
"""


def summarize_profile_batch(profiles: list[dict],
                             use_llm: bool = False) -> list[str]:
    """
    Generate summaries for a list of profile metadata dicts.
    If use_llm=True, uses LLM for richer summaries (slower, costs tokens).
    If use_llm=False, uses rule-based template (fast, free).
    """
    summaries = []
    for p in profiles:
        if use_llm:
            summary = _llm_summary(p)
        else:
            summary = _template_summary(p)
        summaries.append(summary)
    return summaries


def _template_summary(p: dict) -> str:
    """Rule-based summary — fast and free."""
    lat   = p.get("latitude")
    lon   = p.get("longitude")
    basin = p.get("ocean_basin", "Unknown Ocean")
    juld  = str(p.get("juld", ""))[:10]
    t     = p.get("mean_temp")
    s     = p.get("mean_salinity")
    d     = p.get("max_pressure")

    lat_str = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}" if lat is not None else "?"
    lon_str = f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}" if lon is not None else "?"

    parts = [
        f"Argo float {p.get('float_id')} cycle {p.get('cycle_number')}:",
        f"{basin}, {lat_str} {lon_str}, {juld}.",
    ]
    if t is not None:
        parts.append(f"Mean temperature {t:.1f}°C,")
    if s is not None:
        parts.append(f"salinity {s:.2f} PSU,")
    if d is not None:
        parts.append(f"max depth {d:.0f} dbar.")
    return " ".join(parts)


def _llm_summary(p: dict) -> str:
    """LLM-enhanced summary — richer context."""
    client = LLMClient()
    prompt = (
        f"Float ID: {p.get('float_id')}, Cycle: {p.get('cycle_number')}, "
        f"Date: {str(p.get('juld', ''))[:10]}, "
        f"Location: {p.get('latitude'):.2f}°, {p.get('longitude'):.2f}°, "
        f"Basin: {p.get('ocean_basin')}, "
        f"Mean T: {p.get('mean_temp'):.1f}°C, "
        f"Mean S: {p.get('mean_salinity'):.2f} PSU, "
        f"Max depth: {p.get('max_pressure'):.0f} dbar."
    )
    try:
        return client.complete_sync(
            [Message("system", SUMMARY_SYSTEM), Message("user", prompt)],
            temperature=0.3,
            max_tokens=120,
        )
    except Exception as exc:
        logger.warning(f"LLM summary failed, falling back to template: {exc}")
        return _template_summary(p)
