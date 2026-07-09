from abc import ABC, abstractmethod


class EmbeddingClient(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
