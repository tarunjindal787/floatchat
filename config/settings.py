"""
Cloud-aware settings — reads from Streamlit secrets OR .env file.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


def _try_streamlit_secrets() -> dict:
    """Pull secrets from Streamlit Cloud if available."""
    try:
        import streamlit as st
        return dict(st.secrets)
    except Exception:
        return {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────
    app_title: str = "FloatChat"
    app_log_level: str = "INFO"
    max_sql_rows: int = 2000
    rag_top_k: int = 5
    mcp_server_port: int = 8080

    # ── PostgreSQL ───────────────────────────────────────────
    postgres_user: str = "floatchat"
    postgres_password: str = "floatchat_secret"
    postgres_db: str = "argo_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str = "postgresql://floatchat:floatchat_secret@localhost:5432/argo_db"

    # ── ChromaDB ─────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "argo_profiles"
    chroma_in_memory: bool = True   # True = no server needed (Streamlit Cloud)

    # ── LLM ──────────────────────────────────────────────────
    llm_provider: str = "ollama"
    llm_model: str = "mistral"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    together_api_key: str = ""
    together_model: str = "mistralai/Mistral-7B-Instruct-v0.2"

    groq_api_key: str = ""
    groq_model: str = "llama3-70b-8192"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"

    # ── Embeddings ───────────────────────────────────────────
    embedding_provider: str = "local"
    embedding_model: str = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return settings, preferring Streamlit secrets over .env."""
    st_secrets = _try_streamlit_secrets()
    if st_secrets:
        import os
        for k, v in st_secrets.items():
            if k.upper() not in os.environ:
                os.environ[k.upper()] = str(v)
    return Settings()
