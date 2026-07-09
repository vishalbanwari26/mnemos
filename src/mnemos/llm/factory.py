from mnemos.config import Settings, get_settings
from mnemos.llm.base import LLMClient


def get_llm_client(settings: Settings | None = None, *, for_extraction: bool = False) -> LLMClient:
    settings = settings or get_settings()

    if settings.llm_provider == "mock":
        from mnemos.llm.mock_client import MockLLMClient

        return MockLLMClient()

    model = settings.llm_model_extraction if for_extraction else settings.llm_model_chat

    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        from mnemos.llm.anthropic_client import AnthropicLLMClient

        return AnthropicLLMClient(api_key=settings.anthropic_api_key, model=model)

    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        from mnemos.llm.groq_client import GroqLLMClient

        return GroqLLMClient(api_key=settings.groq_api_key, model=model)

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
