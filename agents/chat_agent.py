from agents.base_agent import BaseAgent
from core.llm_provider_factory import get_llm_provider
from logger import logger

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Remember information shared during the current conversation.
"""

_UNAVAILABLE_MESSAGE = (
    "❌ The AI service is currently unavailable.\n\n"
    "Please try again in a few moments."
)


def _turns_to_messages(turns: list[dict]) -> list[dict]:
    messages = []
    for t in turns:
        messages.append({"role": "user", "content": t["query"]})
        messages.append({"role": "assistant", "content": t["response"]})
    return messages


class ChatAgent(BaseAgent):
    """Stateless instance: no conversation memory lives on `self` — the
    caller (Router, via the session-wide Session.turns) passes in the
    conversation history. This is what makes it safe to share one
    ChatAgent instance across concurrent sessions behind a web server."""

    def handle(self, query, turns=None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *_turns_to_messages(turns or []),
            {"role": "user", "content": query},
        ]

        try:
            return get_llm_provider().chat_messages(messages)
        except Exception:
            logger.exception("ChatAgent failed")
            return _UNAVAILABLE_MESSAGE

    def handle_stream(self, query, turns=None, usage=None):
        # usage (core.cost_tracking.Usage) is passed through explicitly
        # rather than via the ambient contextvar core.llm_service/
        # non-streaming chat_messages relies on — confirmed by testing that
        # a contextvars.ContextVar.set() doesn't survive Starlette's
        # per-token threadpool dispatch for streamed responses (see
        # core/llm_provider.py's stream_messages docstring).
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *_turns_to_messages(turns or []),
            {"role": "user", "content": query},
        ]

        chunks = []
        try:
            for chunk in get_llm_provider().stream_messages(messages, usage=usage):
                chunks.append(chunk)
                yield {"type": "token", "content": chunk}
        except Exception:
            logger.exception("ChatAgent streaming failed")
            yield {"type": "done", "response": _UNAVAILABLE_MESSAGE}
            return

        yield {"type": "done", "response": "".join(chunks)}
