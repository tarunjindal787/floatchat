"""
MCP Server — exposes Argo tools via the Model Context Protocol.
Run: python -m backend.mcp_server
"""
from __future__ import annotations

import json
from typing import Any

from loguru import logger

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types as mcp_types
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    logger.warning("mcp package not installed. MCP server disabled.")

from backend.sql_executor import (
    execute_sql, get_float_trajectory,
    get_nearest_floats, get_profile_data, get_bgc_profile,
)
from backend.text_to_sql import translate_to_sql
from ingestion.vector_indexer import semantic_search


if HAS_MCP:
    server = Server("floatchat-mcp")

    # ── Tool: search_profiles ─────────────────────────────────────────────────
    @server.list_tools()
    async def list_tools():
        return [
            mcp_types.Tool(
                name="search_profiles",
                description=(
                    "Semantic search over ARGO profile metadata summaries. "
                    "Use this to find profiles matching a natural language description."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query":  {"type": "string", "description": "Natural language search query"},
                        "top_k":  {"type": "integer", "default": 8},
                        "ocean":  {"type": "string", "description": "Optional ocean basin filter"},
                    },
                    "required": ["query"],
                },
            ),
            mcp_types.Tool(
                name="execute_sql_query",
                description=(
                    "Execute a validated SQL SELECT query against the ARGO PostgreSQL database. "
                    "Returns rows as JSON. Only SELECT statements are allowed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sql":   {"type": "string", "description": "PostgreSQL SELECT query"},
                        "limit": {"type": "integer", "default": 1000},
                    },
                    "required": ["sql"],
                },
            ),
            mcp_types.Tool(
                name="natural_language_to_sql",
                description="Translate a natural language question to a SQL query and execute it.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                    },
                    "required": ["question"],
                },
            ),
            mcp_types.Tool(
                name="get_float_trajectory",
                description="Get the lat/lon/time trajectory for a specific Argo float.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "float_id": {"type": "string", "description": "7-digit WMO float ID"},
                    },
                    "required": ["float_id"],
                },
            ),
            mcp_types.Tool(
                name="get_nearest_floats",
                description="Find the N nearest Argo float profiles to a given lat/lon point.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "latitude":  {"type": "number"},
                        "longitude": {"type": "number"},
                        "top_n":     {"type": "integer", "default": 5},
                    },
                    "required": ["latitude", "longitude"],
                },
            ),
            mcp_types.Tool(
                name="get_ctd_profile",
                description="Get full CTD (temperature/salinity/pressure) profile for a profile_id.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "profile_id": {"type": "integer"},
                    },
                    "required": ["profile_id"],
                },
            ),
            mcp_types.Tool(
                name="get_bgc_profile",
                description="Get BGC data (oxygen, chlorophyll, nitrate) for a profile_id.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "profile_id": {"type": "integer"},
                    },
                    "required": ["profile_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
        try:
            if name == "search_profiles":
                hits = semantic_search(
                    arguments["query"],
                    top_k=arguments.get("top_k", 8),
                )
                result = json.dumps(hits, default=str, indent=2)

            elif name == "execute_sql_query":
                df = execute_sql(arguments["sql"], limit=arguments.get("limit", 1000))
                result = df.to_json(orient="records", date_format="iso")

            elif name == "natural_language_to_sql":
                sql = translate_to_sql(arguments["question"])
                df  = execute_sql(sql)
                result = json.dumps({
                    "sql": sql,
                    "rows": json.loads(df.to_json(orient="records", date_format="iso")),
                }, indent=2)

            elif name == "get_float_trajectory":
                df = get_float_trajectory(arguments["float_id"])
                result = df.to_json(orient="records", date_format="iso")

            elif name == "get_nearest_floats":
                df = get_nearest_floats(
                    arguments["latitude"],
                    arguments["longitude"],
                    top_n=arguments.get("top_n", 5),
                )
                result = df.to_json(orient="records", date_format="iso")

            elif name == "get_ctd_profile":
                df = get_profile_data(arguments["profile_id"])
                result = df.to_json(orient="records", date_format="iso")

            elif name == "get_bgc_profile":
                df = get_bgc_profile(arguments["profile_id"])
                result = df.to_json(orient="records", date_format="iso")

            else:
                result = json.dumps({"error": f"Unknown tool: {name}"})

        except Exception as exc:
            result = json.dumps({"error": str(exc)})

        return [mcp_types.TextContent(type="text", text=result)]

    async def main():
        async with stdio_server() as streams:
            await server.run(*streams, server.create_initialization_options())


if __name__ == "__main__":
    if not HAS_MCP:
        print("Install 'mcp' package to run the MCP server: pip install mcp")
    else:
        import asyncio
        asyncio.run(main())
