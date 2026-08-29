from sentence_transformers import SentenceTransformer

from mnemos.embeddings.base import EmbeddingClient


class SentenceTransformerEmbeddingClient(EmbeddingClient):
    """Local, deterministic embeddings. No API key, no network call, no
    per-request cost or latency variance — keeps the benchmark reproducible.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            # Once cached, load fully offline — skip the Hugging Face Hub
            # "check for updates" HTTP round trips this library otherwise
            # makes on every load, which cost real seconds and contradict
            # "no network call" above.
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except OSError:
            # Not cached yet (first run on this machine) — fall back to a
            # normal load, which downloads and populates the cache.
            self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()
