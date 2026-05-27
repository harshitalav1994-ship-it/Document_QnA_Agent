"""
Application configuration.

All secrets and tunable parameters live here, sourced from environment
variables. Never hardcode keys. A .env file is loaded automatically in
local development; in production these come from the orchestrator.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM provider ---
    # Provider is abstracted so swapping vendors is a one-line change.
    # See services/agent.py::get_llm.
    llm_provider: Literal["groq", "anthropic", "openai"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0

    groq_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # --- Embeddings ---
    # Default to a small local model so the demo runs with zero paid APIs.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Ingestion ---
    max_document_chars: int = Field(default=100_000, description="Hard cap on document text size")
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval ---
    retrieval_k: int = 4

    # --- Observability ---
    langsmith_api_key: str | None = None
    langsmith_project: str = "doc-qa-agent"
    langsmith_tracing: bool = False  # set true to enable LangSmith

    # --- API ---
    api_title: str = "Document Q&A Agent"
    api_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor. Use this everywhere instead of instantiating directly."""
    return Settings()
