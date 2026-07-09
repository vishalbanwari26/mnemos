from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


@dataclass
class ToolCall:
    name: str
    input: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class LLMClient(ABC):
    @abstractmethod
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
        """Return a completion. If `tools` is given and the model invokes one,
        the call(s) are returned in `LLMResponse.tool_calls`."""
        raise NotImplementedError
