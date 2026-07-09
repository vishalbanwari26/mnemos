from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://mnemos:mnemos@localhost:5432/mnemos"

    storage_backend: Literal["postgres", "qdrant", "neo4j"] = "postgres"
    qdrant_local_path: str = "./data/qdrant_local"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "mnemos-neo4j"

    llm_provider: Literal["anthropic", "groq", "mock"] = "anthropic"
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None
    llm_model_chat: str = "claude-sonnet-4-5-20250929"
    llm_model_extraction: str = "claude-sonnet-4-5-20250929"

    embedding_provider: Literal["local", "mock"] = "local"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    retrieval_top_k_semantic: int = 5
    retrieval_top_k_episodic: int = 5
    retrieval_top_n: int = 6
    retrieval_similarity_weight: float = 0.7
    retrieval_recency_weight: float = 0.3
    recency_half_life_days: float = 30.0
    semantic_dedup_threshold: float = 0.92
    extraction_every_n_turns: int = 1

    reflection_merge_threshold: float = 0.80
    reflection_decay_days: float = 14.0
    reflection_decay_factor: float = 0.85
    reflection_forget_confidence_threshold: float = 0.2

    procedural_epsilon: float = 0.15


@lru_cache
def get_settings() -> Settings:
    return Settings()
