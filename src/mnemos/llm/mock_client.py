from collections import deque
from collections.abc import Iterable

from mnemos.llm.base import LLMClient, LLMResponse, Message


class MockLLMClient(LLMClient):
    """Deterministic, offline client for tests and CI.

    Default mode echoes back a short deterministic string derived from the
    latest user message. Tests that need to exercise real parsing logic (e.g.
    extraction's tool-use handling) can instead inject a queue of exact
    responses via `responses=`, consumed in order, one per `complete()` call.
    """

    def __init__(self, responses: Iterable[LLMResponse] | None = None):
        self._scripted: deque[LLMResponse] = deque(responses or [])

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> LLMResponse:
        if self._scripted:
            return self._scripted.popleft()

        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return LLMResponse(
            content=f"[mock] {last_user[:50]}",
            model="mock",
        )
