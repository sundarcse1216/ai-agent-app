from typing import Iterator

import httpx
import ollama
from langsmith import traceable

from config import OLLAMA_MODEL
from core.cost_tracking import record_usage
from core.llm_provider import LLMProvider, log_llm_call
from exception.llm_unavailable_error import LLMUnavailableError

# ollama-python's Client._request_raw catches httpx.ConnectError itself and
# re-raises it as a builtin ConnectionError (see ollama/_client.py) — so we
# catch ConnectionError here, not httpx.ConnectError. Timeouts aren't caught
# internally by the client and propagate as httpx.TimeoutException directly.
# A reachable-but-erroring server (e.g. model not pulled, 5xx) surfaces as
# ollama.ResponseError. All three get normalized to LLMUnavailableError with
# a friendly message so callers get one exception type regardless of which
# provider is configured.
_CONNECTION_ERRORS = (ConnectionError, httpx.TimeoutException)


class OllamaProvider(LLMProvider):

    def __init__(self, model: str = OLLAMA_MODEL):
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

    @traceable(name="OllamaProvider.chat_messages", run_type="llm")
    def chat_messages(self, messages: list[dict], json_mode: bool = False) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                format="json" if json_mode else None,
            )
            content = response["message"]["content"].strip()
            log_llm_call("ollama", messages, content)
            # Ollama is local/free — cost is always $0, but token counts
            # are still tracked for visibility (response.prompt_eval_count/
            # eval_count, confirmed present on the real response object).
            record_usage(response.prompt_eval_count or 0, response.eval_count or 0, 0.0)
            return content
        except _CONNECTION_ERRORS as e:
            raise LLMUnavailableError(
                "The AI service is currently unavailable.\n\n"
                "Please check that Ollama is running and try again."
            ) from e
        except ollama.ResponseError as e:
            if getattr(e, "status_code", None) and e.status_code >= 500:
                raise LLMUnavailableError(
                    "The AI service is currently unavailable.\n\n"
                    "Please try again in a few moments."
                ) from e
            raise

    # Deliberately NOT @traceable: this generator gets consumed through
    # several nested generator layers (ChatAgent.handle_stream ->
    # Router.route_stream -> the SSE route's event_stream) inside
    # Starlette's iterate_in_threadpool. Wrapping it with @traceable
    # reproducibly hung the whole request indefinitely under that
    # combination (confirmed by testing: the exact same call streams fine
    # in a plain synchronous script, only hangs once nested this way
    # through the live HTTP stack) — a LangSmith generator-tracing/
    # threading interaction, not a bug in this code. chat_messages (the
    # non-streaming path, and the classifier's LLM fallback) still gets
    # traced.
    def stream_messages(self, messages: list[dict], usage=None) -> Iterator[str]:
        collected = []
        last_chunk = None
        try:
            for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
                last_chunk = chunk
                content = chunk["message"]["content"]
                if content:
                    collected.append(content)
                    yield content
            # Logged/recorded once, after the stream completes, not per-chunk.
            # The final chunk (done=True) carries the token counts. Recorded
            # into the explicitly-passed `usage` (see LLMProvider.stream_messages'
            # docstring for why this can't use the contextvar record_usage()
            # that chat_messages above uses).
            log_llm_call("ollama", messages, "".join(collected))
            if last_chunk is not None and usage is not None:
                usage.add(last_chunk.prompt_eval_count or 0, last_chunk.eval_count or 0, 0.0)
        except _CONNECTION_ERRORS as e:
            raise LLMUnavailableError(
                "❌ The AI service is currently unavailable.\n\n"
                "Please check that Ollama is running and try again."
            ) from e
        except ollama.ResponseError as e:
            if getattr(e, "status_code", None) and e.status_code >= 500:
                raise LLMUnavailableError(
                    "❌ The AI service is currently unavailable.\n\n"
                    "Please try again in a few moments."
                ) from e
            raise
