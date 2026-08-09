"""
Per-session LLM cost tracking and budget enforcement.

Two different propagation mechanisms are used here, deliberately:

- Non-streaming calls (Router.route(), core/llm_service.py's chat_messages
  path): Usage is recorded via a contextvar (bind_usage(), set by
  Router.route() before calling an agent) rather than threaded through
  every agent/provider method signature explicitly — the same pattern
  OpenTelemetry/Sentry/LangSmith use for propagating "current trace"
  context through arbitrarily deep call stacks. This works here because
  OllamaProvider/OpenAIProvider are process-wide singletons (see
  core/llm_provider_factory.py) shared across concurrent sessions, so usage
  can't be stored as instance state on them without leaking between
  sessions — but a plain, non-generator function call never crosses a
  thread-dispatch boundary partway through, so the contextvar set at the
  top of route() is reliably still active by the time a provider call
  happens deep inside it.
- Streaming calls (Router.route_stream(), stream_messages()): usage is
  passed as an EXPLICIT parameter instead, all the way from Router through
  ChatAgent.handle_stream() to the provider. This is NOT a stylistic
  choice — I tested the ambient/contextvar approach here first and it
  silently failed to record anything. Starlette's iterate_in_threadpool
  (which drives SSE responses) dispatches a *separate* thread-pool call
  per yielded token; a contextvars.ContextVar.set() made in one dispatch
  does not survive into the next one, because each dispatch re-copies
  context from the calling (event-loop) context, not the previous worker
  thread's mutated one. A generator's own local variables/parameters
  (which explicit passing relies on) don't have this problem — they're
  part of the generator's own frame state, restored on every resume
  regardless of which thread executes it.

RAGAgent (agents/rag_agent.py) and ImageAgent (agents/image_agent.py) ARE
covered, despite neither going through LLMProvider:
- RAGAgent's chain calls a LangChain chat model directly. A LangChain
  callback (agents/rag_agent.py's _UsageCallback) hooks its on_llm_end,
  reading message.usage_metadata — confirmed by testing that this field
  (not llm_output['token_usage'], which is OpenAI-only and None for
  ChatOllama) is populated the same way for both providers. The retrieval
  chain fires this callback twice per turn when there's prior history (once
  for query reformulation, once for answer generation) — both get summed.
- ImageAgent's gpt-image-1 response includes its own `usage` field
  (input_tokens/output_tokens) despite being an image endpoint, not a chat
  one — recorded the same way as everything else.

Remaining known gap: the classifier's LLM fallback (core/intent_classifier.py)
runs *before* Router.route()/route_stream() are called, outside the scope
where usage gets bound/passed, so its (rare — only low-confidence queries
trigger it) cost isn't currently counted either.
"""
import contextvars
from dataclasses import dataclass, field

from config import MAX_COST_PER_SESSION_USD

# Approximate OpenAI pricing, USD per 1,000,000 tokens. Verify against
# https://platform.openai.com/pricing before relying on this for real
# budget decisions — prices change and this table can drift out of date;
# it's a best-effort estimate, not billing-accurate.
OPENAI_PRICING_PER_MILLION_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    # gpt-image-1 (agents/image_agent.py): its input_tokens/output_tokens
    # aren't chat tokens, but the same $/1M-token formula applies -- this
    # single blended rate isn't broken out by text-vs-image-token subtype
    # the way OpenAI's own pricing page is, so treat it as more approximate
    # than the chat model rows above.
    "gpt-image-1": {"input": 5.00, "output": 40.00},
}
_FALLBACK_PRICING = {"input": 0.15, "output": 0.60}


def calculate_openai_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = OPENAI_PRICING_PER_MILLION_TOKENS.get(model, _FALLBACK_PRICING)
    return (
        (prompt_tokens / 1_000_000) * pricing["input"]
        + (completion_tokens / 1_000_000) * pricing["output"]
    )


@dataclass
class Usage:
    """Per-session running total. Lives on Session (core/router.py)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0

    def add(self, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost_usd
        self.call_count += 1

    def over_budget(self) -> bool:
        return self.cost_usd >= MAX_COST_PER_SESSION_USD


_current_usage: contextvars.ContextVar["Usage | None"] = contextvars.ContextVar(
    "_current_usage", default=None
)


class bind_usage:
    """Context manager: makes `usage` the ambient accumulator for the
    duration of the block, so any LLMProvider call made within it (however
    deeply nested inside agent code) records into it. Usage outside any
    bound block (e.g. the classifier's own construction, or a call made
    from a context that never entered this) is silently not tracked —
    record_usage() below is a no-op in that case, not an error."""

    def __init__(self, usage: Usage):
        self.usage = usage
        self._token = None

    def __enter__(self):
        self._token = _current_usage.set(self.usage)
        return self.usage

    def __exit__(self, *exc_info):
        _current_usage.reset(self._token)


def record_usage(prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
    """Called by LLMProvider implementations after every successful call."""
    usage = _current_usage.get()
    if usage is not None:
        usage.add(prompt_tokens, completion_tokens, cost_usd)
