from mnemos.config import Settings, get_settings
from mnemos.storage.base import StorageBackend

# Qdrant (embedded, directory-locked) and Neo4j (driver-pooled) need a single
# long-lived instance shared across calls, unlike Postgres which can build a
# cheap wrapper per call. Cached here, keyed by backend name.
_singletons: dict[str, StorageBackend] = {}


def get_storage_backend(settings: Settings | None = None) -> StorageBackend:
    settings = settings or get_settings()
    backend_name = settings.storage_backend

    if backend_name == "postgres":
        from mnemos.storage.postgres_backend import PostgresBackend

        return PostgresBackend()

    if backend_name in _singletons:
        return _singletons[backend_name]

    if backend_name == "qdrant":
        from mnemos.storage.qdrant_backend import QdrantBackend

        backend: StorageBackend = QdrantBackend(path=settings.qdrant_local_path)
    elif backend_name == "neo4j":
        from mnemos.storage.neo4j_backend import Neo4jBackend

        backend = Neo4jBackend(
            uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password
        )
    else:
        raise ValueError(f"Unknown storage backend: {backend_name}")

    _singletons[backend_name] = backend
    return backend


async def reset_storage_backend_cache() -> None:
    """Close and drop cached singleton backends. Used by tests to avoid
    Qdrant's local-mode directory lock leaking across test modules."""
    for backend in _singletons.values():
        await backend.close()
    _singletons.clear()
