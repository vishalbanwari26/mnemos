import hashlib

from mnemos.embeddings.base import EmbeddingClient


class MockEmbeddingClient(EmbeddingClient):
    """Deterministic, hash-seeded pseudo-embeddings for testing retrieval
    *mechanics* (SQL shape, ranking/merge logic) cheaply and offline.

    These vectors carry no real semantic meaning — never use this client for
    the benchmark or for judging retrieval quality.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        # Repeat the 32-byte digest to fill the vector, byte -> [-1, 1) float.
        raw = (digest * (self._dimension // len(digest) + 1))[: self._dimension]
        vector = [(b / 127.5) - 1.0 for b in raw]
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]
