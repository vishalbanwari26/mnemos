from mnemos.llm.base import LLMClient, LLMResponse, Message, ToolCall


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        # Imported lazily so the SDK is only required when this client is used.
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

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
        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        response = await self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(name=block.name, input=block.input))

        return LLMResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )
