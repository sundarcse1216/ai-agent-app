"""
Standalone LangChain learning script — a conversational chat agent built with
LangChain, kept separate from agents/chat_agent.py (which intentionally does
NOT use LangChain — see CLAUDE.md's rationale). Nothing here is imported by
the real app; run this file directly to experiment.

Concepts demonstrated, in order:
1. Chat model selection (same LLM_PROVIDER env var this whole project uses,
   so it works with your existing .env/Ollama setup with no extra config).
2. ChatPromptTemplate + MessagesPlaceholder — how LangChain represents a
   system prompt + conversation history + new user input as one template.
3. LCEL (LangChain Expression Language) — chaining runnables with `|`,
   the idiomatic way to compose prompt -> model -> output-parser.
4. RunnableWithMessageHistory — LangChain's own memory mechanism. This is
   deliberately DIFFERENT from how agents/chat_agent.py and agents/rag_agent.py
   do it in this codebase (they take a plain list[dict] of turns from
   Router/Session and convert it to messages by hand, since Router owns one
   shared history across all six agents). RunnableWithMessageHistory instead
   lets LangChain own a history object per "session_id" you pass it. Seeing
   both approaches side by side is the point of this script.
5. .invoke() vs .stream() — blocking call vs token-by-token generator,
   the same distinction as this project's chat_messages()/stream_messages().
6. Tool calling (see TOOLS/tool_calling_chain below) — binding tools to the
   model with .bind_tools(), and the explicit loop needed to actually run a
   tool call and feed its result back. This step deliberately does NOT use
   a plain `prompt | model | parser` pipe, because "did the model call a
   tool or answer directly?" is a branch, not a straight pipe — see the
   comment on tool_calling_chain for why.

Run it:
    python scripts/langchain_chat_example.py
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory  # noqa: E402
from langchain_core.messages import ToolMessage  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: E402
from langchain_core.runnables import chain  # noqa: E402
from langchain_core.runnables.history import RunnableWithMessageHistory  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from config import LLM_PROVIDER, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_CHAT_MODEL  # noqa: E402

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Remember information shared during the "
    "current conversation. Use the available tools whenever they'd give a "
    "more accurate answer than guessing (e.g. arithmetic, the current time)."
)


# --- Tools -------------------------------------------------------------
# @tool turns an ordinary Python function into a LangChain Tool: it reads
# the function's name, type hints, and docstring to build a schema the
# model is shown (so the model knows the tool exists, what arguments it
# takes, and when the docstring says to use it) — you never hand-write that
# schema yourself.

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


def build_chat_model(bind_tools: bool = False):
    """Same provider-selection idea as agents/rag_agent.py::build_chat_model —
    LangChain needs its own chat-model class, it can't use this project's
    LLMProvider interface directly. bind_tools=True returns a new model
    object with TOOLS attached — .bind_tools() doesn't mutate the model,
    it returns a wrapped copy, so build_chat_model() (no args) still gives
    you a plain, tool-less model when you just want build_chain()'s version."""
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


def build_chain():
    """LCEL: prompt | model | output_parser

    Each `|` feeds the left side's output as the right side's input, exactly
    like a shell pipe. Reading it left to right:
      - `prompt` takes a dict like {"input": "...", "chat_history": [...]}
        and turns it into a list of chat messages.
      - `model` takes that message list and returns an AIMessage.
      - `StrOutputParser()` takes the AIMessage and returns just its
        `.content` string, so callers get a plain str back instead of an
        AIMessage object.

    No tools here — this is the plain version from before tool calling was
    added, kept so you can compare the two side by side.
    """
    return build_prompt() | build_chat_model() | StrOutputParser()


@chain
def tool_calling_chain(input: dict):
    """A tool-aware alternative to build_chain(). Note this is NOT built
    with `|` — it's a plain Python function decorated with @chain, which is
    LCEL's escape hatch: it turns an ordinary function into a Runnable (so
    it still gets .invoke()/.batch(), and works as a drop-in for
    RunnableWithMessageHistory below) while letting you write normal
    if/for control flow inside. `|` alone can't express "call the model,
    then branch on whether it asked for a tool" — piping only ever
    describes a straight line, so once the logic branches, a function is
    the more honest tool. LangGraph exists specifically to make branches
    like this composable/visual once they get more complex than this.

    The actual tool-calling protocol, step by step:
    1. Turn the prompt + this call's input into a message list.
    2. Call the tool-bound model once. If it decides a tool would help
       answering, it returns an AIMessage with .tool_calls populated
       (name + args it wants to call) INSTEAD OF a direct text answer —
       the model does NOT execute anything itself, it only requests it.
    3. We execute each requested tool ourselves (TOOLS_BY_NAME lookup) and
       append the result as a ToolMessage, tagged with tool_call_id so the
       model can match results back to its own requests.
    4. Call the model a second time with the tool results appended, so it
       can produce the final natural-language answer using them.
    If the model didn't call a tool at all (ai_msg.tool_calls is empty),
    steps 3-4 are skipped and we just return its direct answer.
    """
    messages = build_prompt().invoke(input).to_messages()

    model = build_chat_model(bind_tools=True)
    ai_msg = model.invoke(messages)
    messages.append(ai_msg)

    for tool_call in ai_msg.tool_calls:
        result = TOOLS_BY_NAME[tool_call["name"]].invoke(tool_call["args"])
        print(f"\n  [tool call: {tool_call['name']}({tool_call['args']}) -> {result}]")
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    if ai_msg.tool_calls:
        ai_msg = model.invoke(messages)

    return ai_msg.content


# In-memory store of chat_history objects, one per session_id. In a real app
# this would be a database/Redis-backed BaseChatMessageHistory implementation
# instead — this project's own Router avoids this pattern entirely by keeping
# history in api/session.py::SessionStore and passing it in explicitly, which
# is why agents/chat_agent.py has no equivalent of this dict.
_session_histories: dict[str, BaseChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


def build_conversational_chain():
    """Wraps tool_calling_chain so LangChain automatically reads/appends to a
    chat_history for you, keyed by session_id, instead of the caller building
    the message list on every call. input_messages_key/history_messages_key
    tell it which dict keys in your invoke() input correspond to the new
    message vs. where the history should be spliced in (matching the "input"
    and "chat_history" placeholders in build_prompt()). Works the same way
    whether the wrapped runnable is the plain build_chain() pipe or the
    @chain-decorated tool_calling_chain function — RunnableWithMessageHistory
    only cares that the thing it wraps is *some* Runnable, not how it's built."""
    return RunnableWithMessageHistory(
        tool_calling_chain,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def main():
    chain = build_conversational_chain()
    session_id = "demo-session"
    config = {"configurable": {"session_id": session_id}}

    print(f"LangChain chat example (provider={LLM_PROVIDER}). Tools: add, get_current_time. Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        # .invoke() here, not .stream() — tool_calling_chain may need to call
        # the model twice (once to see if a tool is requested, once more
        # after running it) before it has a final answer, so there's no
        # single continuous token stream to yield mid-way through that.
        # Contrast with build_chain()'s plain prompt|model|parser pipe,
        # which streams fine (see the earlier version of this script /
        # git history) because it's always exactly one model call.
        answer = chain.invoke({"input": query}, config=config)
        print(f"AI: {answer}\n")


if __name__ == "__main__":
    main()
