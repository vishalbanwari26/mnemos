from sentence_transformers import SentenceTransformer

from mnemos.embeddings.base import EmbeddingClient


class SentenceTransformerEmbeddingClient(EmbeddingClient):
    """Local, deterministic embeddings. No API key, no network call, no
    per-request cost or latency variance — keeps the benchmark reproducible.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()
