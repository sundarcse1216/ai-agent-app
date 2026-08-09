"""
Thin backwards-compatible facade over the configured LLMProvider.

Kept so WeatherAgent, RecommendationAgent, and SQLAgent (and their existing
tests, which mock LLMService.chat) can keep calling
LLMService.chat(system_prompt, user_prompt, json_mode=...) unchanged while
the underlying LLM backend (Ollama vs. OpenAI) becomes swappable via
core.llm_provider_factory.
"""
from core.llm_provider_factory import get_llm_provider
from exception.llm_unavailable_error import LLMUnavailableError
from logger import logger


class LLMService:

    @staticmethod
    def chat(system_prompt: str, user_prompt: str, json_mode=False):
        try:
            return get_llm_provider().chat(system_prompt, user_prompt, json_mode=json_mode)
        except LLMUnavailableError:
            logger.exception("LLM unavailable")
            raise
        except Exception as e:
            logger.exception(f"LLM call failed: {e}")
            raise
