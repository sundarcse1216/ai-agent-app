import functools

from config import LLM_PROVIDER
from core.llm_provider import LLMProvider


@functools.lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Returns the configured LLMProvider singleton (built once, reused for
    the process lifetime — provider construction sets up an SDK client)."""

    if LLM_PROVIDER == "ollama":
        from core.providers.ollama_provider import OllamaProvider
        return OllamaProvider()

    if LLM_PROVIDER == "openai":
        from core.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()

    raise ValueError(
        f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}. Expected 'ollama' or 'openai'."
    )
