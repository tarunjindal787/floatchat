"""
FloatChat Application Settings
Uses pydantic-settings for typed, env-file-driven configuration.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────
    app_title: str = "FloatChat"
    app_log_level: str = "INFO"
    max_sql_rows: int = 5000
    rag_top_k: int = 8
    mcp_server_port: int = 8080

    # ── PostgreSQL ───────────────────────────────────────────
    postgres_user: str = "floatchat"
    postgres_password: str = "floatchat_secret"
    postgres_db: str = "argo_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str = (
        "postgresql://floatchat:floatchat_secret@localhost:5432/argo_db"
    )

    # ── ChromaDB ─────────────────────────────────────────────
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "argo_profiles"

    # ── LLM ──────────────────────────────────────────────────
    llm_provider: str = "ollama"          # openai | ollama | together | groq
    llm_model: str = "mistral"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    together_api_key: str = ""
    together_model: str = "mistralai/Mistral-7B-Instruct-v0.2"

    groq_api_key: str = ""
    groq_model: str = "llama3-70b-8192"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"

    # ── Embeddings ───────────────────────────────────────────
    embedding_provider: str = "local"     # local | openai
    embedding_model: str = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
