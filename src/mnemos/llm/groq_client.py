import json

from mnemos.llm.base import LLMClient, LLMResponse, Message, ToolCall


def _to_openai_tool(tool: dict) -> dict:
    """extraction.py/reflection.py define tools in Anthropic's shape
    ({"name", "description", "input_schema"}) since Anthropic was the first
    provider. Groq's API is OpenAI-compatible, which wants
    {"type": "function", "function": {"name", "description", "parameters"}} —
    translated here so the call sites stay provider-agnostic."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool["input_schema"],
        },
    }


def _to_openai_tool_choice(tool_choice: dict) -> dict | str:
    if tool_choice.get("type") == "tool" and "name" in tool_choice:
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return "auto"


class GroqLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        # Imported lazily so the SDK is only required when this client is used.
        from groq import AsyncGroq

        self._client = AsyncGroq(api_key=api_key)
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
        groq_messages: list[dict] = []
        if system:
            groq_messages.append({"role": "system", "content": system})
        groq_messages.extend({"role": m["role"], "content": m["content"]} for m in messages)

        kwargs: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": groq_messages,
        }
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]
        if tool_choice:
            kwargs["tool_choice"] = _to_openai_tool_choice(tool_choice)

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        tool_calls = [
            ToolCall(name=tc.function.name, input=json.loads(tc.function.arguments))
            for tc in (choice.tool_calls or [])
        ]

        return LLMResponse(
            content=choice.content or "",
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=response.model,
        )
