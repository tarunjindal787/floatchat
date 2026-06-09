"""
Response Generator — builds the final LLM response combining SQL results, RAG context,
and chart specifications.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from loguru import logger
from tabulate import tabulate

from backend.models.llm_client import LLMClient, Message
from backend.query_router import QueryIntent, RouterResult, route_query
from backend.text_to_sql import translate_to_sql
from backend.sql_executor import (
    execute_sql, get_float_trajectory,
    get_nearest_floats, get_profile_data,
)
from backend.rag_retriever import retrieve_context, format_context


@dataclass
class FloatChatResponse:
    text: str
    data: pd.DataFrame | None = None
    chart_type: str | None = None          # trajectory | profile | heatmap | bgc | ts
    chart_config: dict = field(default_factory=dict)
    sql_used: str | None = None
    sources: list[dict] = field(default_factory=list)
    error: str | None = None


SYSTEM_PERSONA = """You are FloatChat, an expert AI assistant specializing in ARGO oceanographic float data.
You help scientists, decision-makers, and students explore ocean temperature, salinity, and BGC parameters.
Be concise, accurate, and scientifically informative. When presenting data:
- Highlight key insights and anomalies.
- Mention data quality considerations (QC flags).
- Suggest follow-up analyses when appropriate.
- Always cite float IDs, dates, and regions in your answer.
"""


class ResponseGenerator:

    def __init__(self):
        self.llm = LLMClient()

    def generate(self, user_query: str,
                 history: list[dict] | None = None) -> FloatChatResponse:
        """
        Main entry point. Routes the query and returns a FloatChatResponse.
        """
        history = history or []
        route = route_query(user_query)
        logger.info(f"Intent: {route.intent} | Confidence: {route.confidence:.2f}")

        try:
            if route.intent == QueryIntent.TRAJECTORY:
                return self._handle_trajectory(user_query, route, history)
            elif route.intent == QueryIntent.NEAREST:
                return self._handle_nearest(user_query, route, history)
            elif route.intent == QueryIntent.SQL:
                return self._handle_sql(user_query, route, history)
            elif route.intent == QueryIntent.RAG:
                return self._handle_rag(user_query, history)
            elif route.intent == QueryIntent.EXPORT:
                return self._handle_export(user_query, route)
            else:
                return self._handle_general(user_query, history)
        except Exception as exc:
            logger.exception(f"Error generating response: {exc}")
            return FloatChatResponse(
                text="I encountered an error processing your request. Please try rephrasing.",
                error=str(exc),
            )

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_sql(self, query: str, route: RouterResult,
                    history: list[dict]) -> FloatChatResponse:
        sql = translate_to_sql(query)
        df  = execute_sql(sql)

        context = _build_data_context(df)
        reply   = self._llm_reply(query, context, history)

        chart_type, chart_cfg = _infer_chart(query, df)

        return FloatChatResponse(
            text=reply,
            data=df,
            sql_used=sql,
            chart_type=chart_type,
            chart_config=chart_cfg,
        )

    def _handle_trajectory(self, query: str, route: RouterResult,
                            history: list[dict]) -> FloatChatResponse:
        float_ids = route.entities.get("float_ids", [])
        if not float_ids:
            return FloatChatResponse(
                text="Please specify a float ID (7-digit number) to show its trajectory.",
            )
        float_id = float_ids[0]
        df = get_float_trajectory(float_id)

        context = f"Float {float_id} trajectory: {len(df)} positions from {df['juld'].min()} to {df['juld'].max()}."
        reply = self._llm_reply(query, context, history)

        return FloatChatResponse(
            text=reply,
            data=df,
            chart_type="trajectory",
            chart_config={"float_id": float_id, "color_by": "juld"},
        )

    def _handle_nearest(self, query: str, route: RouterResult,
                         history: list[dict]) -> FloatChatResponse:
        lat = route.entities.get("latitude")
        lon = route.entities.get("longitude")
        df  = get_nearest_floats(lat, lon, top_n=10)

        context = f"Nearest floats to ({lat}, {lon}):\n{df.to_string(index=False)}"
        reply   = self._llm_reply(query, context, history)

        return FloatChatResponse(
            text=reply,
            data=df,
            chart_type="heatmap",
            chart_config={"center_lat": lat, "center_lon": lon},
        )

    def _handle_rag(self, query: str, history: list[dict]) -> FloatChatResponse:
        hits    = retrieve_context(query)
        context = format_context(hits)
        reply   = self._llm_reply(query, context, history)

        return FloatChatResponse(
            text=reply,
            sources=[h["metadata"] for h in hits],
        )

    def _handle_export(self, query: str, route: RouterResult) -> FloatChatResponse:
        float_ids = route.entities.get("float_ids", [])
        if not float_ids:
            return FloatChatResponse(
                text="Specify a float ID to export data. Example: 'Export float 6902742 as NetCDF'",
            )
        sql = f"SELECT * FROM profile_stats WHERE float_id = '{float_ids[0]}'"
        df  = execute_sql(sql)
        return FloatChatResponse(
            text=f"Data for float {float_ids[0]} ready for download ({len(df)} profiles).",
            data=df,
            sql_used=sql,
        )

    def _handle_general(self, query: str, history: list[dict]) -> FloatChatResponse:
        reply = self._llm_reply(query, "", history)
        return FloatChatResponse(text=reply)

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _llm_reply(self, query: str, context: str, history: list[dict]) -> str:
        messages = [Message("system", SYSTEM_PERSONA)]
        for h in history[-6:]:  # keep last 6 turns
            messages.append(Message(h["role"], h["content"]))
        if context:
            messages.append(Message("system", f"Data context:\n{context}"))
        messages.append(Message("user", query))
        return self.llm.complete_sync(messages)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _build_data_context(df: pd.DataFrame) -> str:
    if df.empty:
        return "The query returned no results."
    summary = df.describe(include="all").to_string()
    sample  = tabulate(df.head(5), headers="keys", tablefmt="plain", showindex=False)
    return f"Query returned {len(df)} rows.\n\nSample:\n{sample}\n\nStats:\n{summary}"


def _infer_chart(query: str, df: pd.DataFrame) -> tuple[str | None, dict]:
    q = query.lower()
    cols = set(df.columns)

    if "pressure" in cols and ("temperature" in cols or "salinity" in cols):
        return "profile", {"x_col": "temperature", "y_col": "pressure"}
    if "latitude" in cols and "longitude" in cols:
        return "heatmap", {}
    if "juld" in cols and any(c in cols for c in ["temperature", "salinity", "doxy", "chla"]):
        return "timeseries", {}
    return None, {}
