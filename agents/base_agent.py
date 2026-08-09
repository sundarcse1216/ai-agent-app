from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def handle(self, query: str):
        pass

    def handle_stream(self, query: str, turns=None):
        """Optional: agents that can produce incremental output (currently
        ChatAgent, RAGAgent) override this as a generator yielding
        {"type": "token", "content": str} chunks followed by exactly one
        {"type": "done", "response": str}. Agents that do one or more
        blocking calls with no meaningful partial output (weather/SQL/
        recommendation/image) don't override it — the default raises, and
        the caller (Router.route_stream) falls back to a single blocking
        handle() call emitted as one event instead of faking a token
        stream."""
        raise NotImplementedError
