"""
One shared conversation history per Session, used by every context-aware
agent (ChatAgent, RAGAgent, WeatherAgent, RecommendationAgent) regardless
of which of them handled a given turn.

This replaces per-agent history (a prior version of this app kept separate
chat_history/rag_history lists) — that meant "my name is Sundar" said
during a query that happened to route to RAG was invisible to ChatAgent on
the next turn, and vice versa, even though the user experiences this as
one continuous conversation. Router.route()/route_stream() call
record_turn() after every turn, for every intent, so any later turn (by
any agent) can see what was said earlier regardless of which agent said it.

Each stateful agent converts session.turns into whatever message format
its own call path needs (ChatAgent/RAGAgent: LangChain HumanMessage/
AIMessage). WeatherAgent/RecommendationAgent just read the raw query text
out of it for single-slot extraction context (see format_recent_turns).
"""
from config import MEMORY_WINDOW
from logger import logger


def format_recent_turns(turns: list[dict] | None) -> str:
    if not turns:
        return ""
    lines = "\n".join(f'- "{t["query"]}"' for t in turns)
    return f"\nRecent conversation (oldest to newest), for context only:\n{lines}\n"


def record_turn(session, query: str, response: str, agent: str) -> None:
    # Centralized agent-response tracking: this is called for every turn,
    # regardless of intent, so it's the one place that sees every agent's
    # final answer — no need to add logging to each agent individually.
    logger.info(f"Agent response | agent={agent} query={query!r} response={response!r}")

    session.turns.append({"query": query, "response": response, "agent": agent})
    if len(session.turns) > MEMORY_WINDOW:
        session.turns = session.turns[-MEMORY_WINDOW:]
