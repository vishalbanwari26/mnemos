from mnemos.config import Settings, get_settings
from mnemos.embeddings.base import EmbeddingClient

_cached_client: EmbeddingClient | None = None


def get_embedding_client(settings: Settings | None = None) -> EmbeddingClient:
    """Cached for the default settings (the model load is ~1-2s); a custom
    `settings` bypasses the cache and always builds a fresh client.
    """
    global _cached_client
    use_default = settings is None
    if use_default and _cached_client is not None:
        return _cached_client

    settings = settings or get_settings()

    if settings.embedding_provider == "mock":
        from mnemos.embeddings.mock_client import MockEmbeddingClient

        client: EmbeddingClient = MockEmbeddingClient(dimension=settings.embedding_dimension)
    elif settings.embedding_provider == "local":
        from mnemos.embeddings.local_client import SentenceTransformerEmbeddingClient

        client = SentenceTransformerEmbeddingClient(model_name=settings.embedding_model_name)
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")

    if client.dimension != settings.embedding_dimension:
        raise ValueError(
            f"Embedding client dimension ({client.dimension}) does not match "
            f"settings.embedding_dimension ({settings.embedding_dimension}). "
            "The pgvector columns are fixed-width — update EMBEDDING_DIMENSION "
            "and the schema together if you change the model."
        )
    if use_default:
        _cached_client = client
    return client
