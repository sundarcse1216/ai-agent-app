"""
LLM provider abstraction.

Every agent that needs a raw chat-completion call (as opposed to a full
LangChain chain, like RAGAgent's retrieval chain) goes through an
LLMProvider instead of talking to Ollama/OpenAI directly. This is what
makes the LLM backend swappable via the LLM_PROVIDER config value without
touching agent code.
"""
from abc import ABC, abstractmethod
from typing import Iterator

from logger import logger


def log_llm_call(provider: str, messages: list[dict], response: str) -> None:
    """Shared by both provider implementations so every successful LLM
    call — the request messages and the full response text — lands in
    outputs/logs/app.log, regardless of which agent triggered it or which
    provider is configured. Failures are already logged separately at the
    point they're raised (LLMUnavailableError etc.)."""
    logger.info(f"LLM call | provider={provider} messages={messages} response={response!r}")


class LLMProvider(ABC):

    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        """Single system+user turn, returns the full response text."""

    @abstractmethod
    def stream(self, system_prompt: str, user_prompt: str) -> Iterator[str]:
        """Single system+user turn, yields response text chunks as they arrive."""

    @abstractmethod
    def chat_messages(self, messages: list[dict], json_mode: bool = False) -> str:
        """Multi-turn call: messages is a list of {"role", "content"} dicts
        (roles: system/user/assistant), for agents carrying conversation history.
        Records cost into core.cost_tracking's ambient (contextvar-bound)
        Usage accumulator — safe here because this is a single, non-generator
        call (see stream_messages for why streaming can't use the same trick)."""

    @abstractmethod
    def stream_messages(self, messages: list[dict], usage=None) -> Iterator[str]:
        """Streaming variant of chat_messages. Takes an explicit `usage:
        core.cost_tracking.Usage | None` parameter instead of the ambient
        contextvar chat_messages uses — confirmed by direct testing that
        Starlette's iterate_in_threadpool dispatches a *separate* thread-pool
        call per yielded token, and a contextvars.ContextVar.set() made in
        one dispatch does not survive into the next one. Ambient context
        only works within a single, non-yielding call; a generator consumed
        this way needs state passed explicitly instead."""
