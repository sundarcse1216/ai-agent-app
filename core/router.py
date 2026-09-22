from dataclasses import dataclass, field

from agents.chat_agent import ChatAgent
from agents.image_agent import ImageAgent
from agents.rag_agent import RAGAgent
from agents.recommendation_system import RecommendationAgent
from agents.sql_agent import SQLAgent
from agents.weather_agent import WeatherAgent
from core.conversation_context import record_turn
from core.cost_tracking import Usage, bind_usage

# Intents whose agent takes the shared conversation history (session.turns)
# as context — either as full conversational memory (chat, rag) or to
# recover a slot like location/date mentioned in an earlier turn handled
# by a *different* agent (weather, recommendation). Every other intent's
# agent.handle(query) is a single self-contained call with no context.
CONTEXT_AWARE_INTENTS = {"chat", "rag", "weather", "recommendation"}

# Of those, only these implement handle_stream (chat/rag do real token
# streaming; weather/recommendation are single blocking LLM calls with no
# meaningful partial output, so they always go through the non-streaming
# route() fallback in route_stream() below).
STREAMABLE_INTENTS = {"chat", "rag"}

_BUDGET_EXCEEDED_MESSAGE = (
    "This session has reached its usage limit."
)


@dataclass
class Session:
    """Per-session state: shared conversation history (turns) plus a
    running LLM cost total (usage) enforced as a hard per-session budget.

    turns — every turn's query+response, regardless of which agent handled
    it (see core/conversation_context.py for why this is a single list
    rather than one per agent).

    usage — see core/cost_tracking.py. For non-streaming calls (route()),
    bound as ambient context via bind_usage() so LLMProvider implementations
    (process-wide singletons shared across concurrent sessions) can record
    cost into the CORRECT session without Router threading a reference
    through every agent method signature. For streaming calls
    (route_stream()), that ambient approach doesn't work (confirmed by
    testing — see core/llm_provider.py's stream_messages docstring), so
    usage is instead passed explicitly into handle_stream().

    Everything else about an agent — its LLM client, FAISS index, retrieval
    chain — is pooled and stateless-across-sessions, built once on the
    agent instance itself in Router.__init__. Only the state above is
    per-session, so it lives here instead of on the agent.

    Controller owns exactly one Session today (a console REPL is one
    implicit conversation). The web layer (api/session.py) keeps one
    Session per client, looked up by session id."""
    turns: list = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class Router:

    def __init__(self):
        # Agents are built lazily, on first use of their intent, instead of
        # all six eagerly here. RAGAgent alone (embedding model + FAISS
        # index) combined with the classifier's own embedding model was
        # enough to OOM a 512MB-RAM deploy target (e.g. Render's free tier)
        # before the app ever served a request — confirmed live, not
        # theoretical. Lazily building means a deploy that never gets a RAG
        # question never pays RAGAgent's memory cost. Each agent is still
        # built at most once and reused for the process lifetime (same
        # pooled, stateless-across-sessions contract as before) — this
        # defers construction, it doesn't rebuild per request.
        self._agent_factories = {
            "weather": lambda: WeatherAgent(),
            "sql": lambda: SQLAgent(),
            "recommendation": lambda: RecommendationAgent(self._get_agent("weather")),
            "rag": lambda: RAGAgent(),
            "image": lambda: ImageAgent(),
            "chat": lambda: ChatAgent(),
        }
        self.agents = {}

    def _get_agent(self, intent):
        if intent not in self.agents:
            factory = self._agent_factories.get(intent)
            if factory is None:
                return None
            self.agents[intent] = factory()
        return self.agents[intent]

    def route(self, intent, query, session: Session):
        agent = self._get_agent(intent)

        if not agent:
            return "Sorry, I couldn't determine the appropriate agent."

        if session.usage.over_budget():
            return _BUDGET_EXCEEDED_MESSAGE

        with bind_usage(session.usage):
            if intent in ("rag", "chat", "weather"):
                # rag/chat/weather all need usage passed explicitly — none
                # of them go through LLMProvider anymore, so each calls a
                # LangChain chat model directly and tracks cost via its own
                # LangChain callback (agents/rag_agent.py's /
                # agents/chat_agent.py's / agents/weather_agent.py's
                # _UsageCallback), not LLMProvider's ambient record_usage()
                # the other agents (sql, recommendation, image) use.
                response = agent.handle(query, session.turns, usage=session.usage)
            elif intent in CONTEXT_AWARE_INTENTS:
                response = agent.handle(query, session.turns)
            else:
                response = agent.handle(query)

        record_turn(session, query, response, intent)
        return response

    def route_stream(self, intent, query, session: Session):
        """Streaming counterpart to route(). Yields {"type": "token", ...}
        events followed by exactly one {"type": "result", "response": ...}.
        Non-streamable agents (and any streamable agent that doesn't
        implement handle_stream) fall back to a single blocking route()
        call emitted as one result event — same context-threading logic
        either way, just via BaseAgent.handle_stream's NotImplementedError
        as the signal to fall back."""
        agent = self._get_agent(intent)

        if not agent:
            yield {"type": "result", "response": "Sorry, I couldn't determine the appropriate agent."}
            return

        if session.usage.over_budget():
            yield {"type": "result", "response": _BUDGET_EXCEEDED_MESSAGE}
            return

        if intent in STREAMABLE_INTENTS:
            try:
                response_text = None
                # usage is passed explicitly here, NOT via bind_usage()
                # (used below in route() for the non-streaming path) —
                # confirmed by direct testing that a contextvar set here
                # would not survive Starlette's per-token threadpool
                # dispatch for this generator (see core/llm_provider.py's
                # stream_messages docstring for the full explanation).
                for event in agent.handle_stream(query, session.turns, usage=session.usage):
                    if event["type"] == "token":
                        yield {"type": "token", "content": event["content"]}
                    else:
                        response_text = event["response"]
                record_turn(session, query, response_text, intent)
                yield {"type": "result", "response": response_text}
                return
            except NotImplementedError:
                pass  # agent doesn't support streaming -- fall through below

        # route() checks budget/records the turn/binds usage itself.
        yield {"type": "result", "response": self.route(intent, query, session)}
