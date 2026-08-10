from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from config import LLM_PROVIDER, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_CHAT_MODEL
from core.cost_tracking import calculate_openai_cost
from logger import logger

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Remember information shared during the current conversation. Use the
available tools whenever they'd give a more accurate answer than guessing
(e.g. arithmetic, the current time).
"""

_UNAVAILABLE_MESSAGE = (
    "The AI service is currently unavailable.\n\n"
    "Please try again in a few moments."
)


# --- Tools ---------------------------------------------------------------
# @tool turns a plain function into something the model can be shown a
# schema for (name/args/docstring) and request a call to — the model only
# ever *requests* a call (via .tool_calls on its response); we're the ones
# who actually execute it below and feed the result back.

@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the exact result."""
    return a + b


@tool
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [add, get_current_time]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class _UsageCallback(BaseCallbackHandler):
    """Same approach as agents/rag_agent.py's _UsageCallback (duplicated
    rather than shared — each agent file in this codebase is self-contained,
    e.g. RAGAgent has its own build_chat_model() too). Reads
    message.usage_metadata rather than llm_output['token_usage'], since the
    latter is confirmed None for ChatOllama (langchain_ollama doesn't
    populate it) though present for ChatOpenAI; usage_metadata is populated
    the same way for both. Passed in explicitly via config={"callbacks":[...]}
    rather than core.cost_tracking's ambient contextvar, since this callback
    fires from inside LangChain's own execution machinery — Router already
    established (see core/cost_tracking.py's module docstring) that ambient
    context isn't reliable across anything but a single, non-generator,
    non-nested call."""

    def __init__(self, usage, model: str, is_openai: bool):
        self.usage = usage
        self.model = model
        self.is_openai = is_openai

    def on_llm_end(self, response, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                message = getattr(gen, "message", None)
                usage_metadata = getattr(message, "usage_metadata", None) if message else None
                if not usage_metadata:
                    continue
                prompt_tokens = usage_metadata.get("input_tokens", 0)
                completion_tokens = usage_metadata.get("output_tokens", 0)
                cost = (
                    calculate_openai_cost(self.model, prompt_tokens, completion_tokens)
                    if self.is_openai else 0.0
                )
                self.usage.add(prompt_tokens, completion_tokens, cost)


def _usage_config(usage) -> dict:
    if usage is None:
        return {}
    return {"callbacks": [_UsageCallback(usage, OPENAI_CHAT_MODEL, LLM_PROVIDER == "openai")]}


def build_chat_model(bind_tools: bool = False):
    """Same provider-selection idea as agents/rag_agent.py::build_chat_model —
    LangChain needs its own chat-model class, it can't use this project's
    LLMProvider interface directly. bind_tools=True returns a new model
    object with TOOLS attached (.bind_tools() returns a wrapped copy, it
    doesn't mutate the model in place)."""
    if LLM_PROVIDER == "openai":
        model = ChatOpenAI(model=OPENAI_CHAT_MODEL, api_key=OPENAI_API_KEY)
    else:
        model = ChatOllama(model=OLLAMA_MODEL)
    return model.bind_tools(TOOLS) if bind_tools else model


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


def build_chain(bind_tools: bool = False):
    """LCEL: prompt | model. The same shape as agents/rag_agent.py's chain
    builders — create_history_aware_retriever/create_stuff_documents_chain
    are themselves prompt|model pipes under the hood for the retrieval
    case; this is the equivalent for a plain chat agent with no retrieval
    step, built explicitly with `|` instead of a prebuilt helper since
    there's no equivalent helper for "just chat"."""
    return build_prompt() | build_chat_model(bind_tools=bind_tools)


def _turns_to_messages(turns: list[dict]) -> list:
    messages = []
    for t in turns:
        messages.append(HumanMessage(content=t["query"]))
        messages.append(AIMessage(content=t["response"]))
    return messages


def _run_tool_calls(ai_msg, messages: list) -> None:
    """Executes every tool the model requested and appends each result as a
    ToolMessage (tagged with tool_call_id so the model can match results
    back to its own requests) — mutates `messages` in place so both
    handle()/handle_stream() can share this."""
    for tool_call in ai_msg.tool_calls:
        result = TOOLS_BY_NAME[tool_call["name"]].invoke(tool_call["args"])
        logger.info(f"ChatAgent tool call: {tool_call['name']}({tool_call['args']}) -> {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))


class ChatAgent(BaseAgent):
    """Stateless instance: no conversation memory lives on `self` — the
    caller (Router, via the session-wide Session.turns) passes in the
    conversation history. This is what makes it safe to share one
    ChatAgent instance across concurrent sessions behind a web server."""

    def handle(self, query, turns=None, usage=None):
        chain_input = {"input": query, "chat_history": _turns_to_messages(turns or [])}

        try:
            ai_msg = build_chain(bind_tools=True).invoke(chain_input, config=_usage_config(usage))

            if ai_msg.tool_calls:
                # A ToolMessage doesn't fit build_prompt()'s ("system",
                # chat_history, "human") template shape, so the tool round
                # trip drops down to a raw message list built from that same
                # template rather than trying to force it back through the
                # prompt|model pipe — see build_chain()'s docstring on why
                # LCEL piping only covers the single-call case here.
                messages = build_prompt().invoke(chain_input).to_messages()
                messages.append(ai_msg)
                _run_tool_calls(ai_msg, messages)
                # Second call lets the model turn tool results into a
                # natural-language answer; it may legitimately choose not to
                # call anything further, so no loop here — one round trip
                # of tool calls per user message is enough for TOOLS above.
                ai_msg = build_chat_model(bind_tools=True).invoke(messages, config=_usage_config(usage))

            return ai_msg.content
        except Exception:
            logger.exception("ChatAgent failed")
            return _UNAVAILABLE_MESSAGE

    def handle_stream(self, query, turns=None, usage=None):
        # usage (core.cost_tracking.Usage) is passed through explicitly
        # rather than via the ambient contextvar core.cost_tracking's
        # bind_usage() relies on — confirmed by testing that a
        # contextvars.ContextVar.set() doesn't survive Starlette's
        # per-token threadpool dispatch for streamed responses (see
        # core/cost_tracking.py's module docstring).
        chain_input = {"input": query, "chat_history": _turns_to_messages(turns or [])}

        try:
            # Stream the first pass through the prompt|model chain and
            # accumulate it into one aggregated AIMessageChunk (chunks
            # support `+`) so we can inspect .tool_calls once the stream
            # ends — this avoids a wasted blocking round trip just to check
            # "did it want a tool" before ever streaming anything, which is
            # what a naive invoke-then-maybe-stream design would cost on
            # every single message, tool or not.
            full = None
            for chunk in build_chain(bind_tools=True).stream(chain_input, config=_usage_config(usage)):
                if chunk.content:
                    yield {"type": "token", "content": chunk.content}
                full = chunk if full is None else full + chunk

            if not full.tool_calls:
                yield {"type": "done", "response": full.content}
                return

            # A tool call produces no (or empty) streamed content on this
            # first pass, so nothing user-visible was emitted above yet.
            # Same reasoning as handle(): a ToolMessage doesn't fit the
            # prompt template's shape, so this drops to a raw message list.
            messages = build_prompt().invoke(chain_input).to_messages()
            messages.append(full)
            _run_tool_calls(full, messages)

            model = build_chat_model(bind_tools=True)

            chunks = []
            for chunk in model.stream(messages, config=_usage_config(usage)):
                if chunk.content:
                    chunks.append(chunk.content)
                    yield {"type": "token", "content": chunk.content}

            yield {"type": "done", "response": "".join(chunks)}
        except Exception:
            logger.exception("ChatAgent streaming failed")
            yield {"type": "done", "response": _UNAVAILABLE_MESSAGE}
