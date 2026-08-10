from typing import Iterator

import openai
from langsmith import traceable
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL
from core.cost_tracking import calculate_openai_cost, record_usage
from core.llm_provider import LLMProvider, log_llm_call
from exception.llm_unavailable_error import LLMUnavailableError

# Errors that mean "the service itself is unreachable/overloaded" rather than
# "the request was bad" — these get normalized to LLMUnavailableError so
# callers get one exception type regardless of which provider is configured.
_UNAVAILABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class OpenAIProvider(LLMProvider):

    def __init__(self, model: str = OPENAI_CHAT_MODEL):
        if not OPENAI_API_KEY:
            raise LLMUnavailableError(
                "❌ The AI service is currently unavailable.\n\n"
                "OPENAI_API_KEY is not configured."
            )
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    @staticmethod
    def _to_messages(system_prompt: str, user_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        return self.chat_messages(
            self._to_messages(system_prompt, user_prompt),
            json_mode=json_mode,
        )

    def stream(self, system_prompt: str, user_prompt: str, usage=None) -> Iterator[str]:
        yield from self.stream_messages(self._to_messages(system_prompt, user_prompt), usage=usage)

    @traceable(name="OpenAIProvider.chat_messages", run_type="llm")
    def chat_messages(self, messages: list[dict], json_mode: bool = False) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"} if json_mode else None,
            )
            content = response.choices[0].message.content.strip()
            log_llm_call("openai", messages, content)
            usage = response.usage
            if usage is not None:
                cost = calculate_openai_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
                record_usage(usage.prompt_tokens, usage.completion_tokens, cost)
            return content
        except _UNAVAILABLE_ERRORS as e:
            raise LLMUnavailableError(
                "The AI service is currently unavailable.\n\n"
                "Please try again in a few moments."
            ) from e

    # Deliberately NOT @traceable — see the matching note in
    # ollama_provider.py: tracing a generator consumed through this app's
    # nested streaming layers (ChatAgent -> Router -> the SSE route) inside
    # Starlette's iterate_in_threadpool reproducibly hung the request.
    def stream_messages(self, messages: list[dict], usage=None) -> Iterator[str]:
        collected = []
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                # The final chunk of a stream_options={"include_usage": True}
                # stream carries usage but an EMPTY choices list — guard
                # against that below rather than assuming choices[0] exists.
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        collected.append(delta)
                        yield delta
                # Recorded into the explicitly-passed `usage` — see
                # LLMProvider.stream_messages' docstring for why this can't
                # use the contextvar record_usage() chat_messages uses above.
                if chunk.usage is not None and usage is not None:
                    cost = calculate_openai_cost(
                        self.model, chunk.usage.prompt_tokens, chunk.usage.completion_tokens
                    )
                    usage.add(chunk.usage.prompt_tokens, chunk.usage.completion_tokens, cost)
            # Logged once, after the stream completes, not per-chunk.
            log_llm_call("openai", messages, "".join(collected))
        except _UNAVAILABLE_ERRORS as e:
            raise LLMUnavailableError(
                "❌ The AI service is currently unavailable.\n\n"
                "Please try again in a few moments."
            ) from e
