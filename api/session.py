"""
Per-X-Session-Id conversation state, so concurrent browser sessions hitting
the one pooled Router instance (built once at app startup) don't bleed
chat/RAG history into each other — see core.router.Session for why this is
needed at all (agents are stateless; the caller owns conversation history).

Single-process, in-memory (TTLCache — reuses the same eviction idiom as
weather_agent's/rag_agent's per-response caches). This is a deliberate
scope boundary for "lean production, no auth, single instance": a
multi-worker/multi-instance deployment would need this moved to something
shared like Redis, since each worker process would otherwise have its own
independent session dict.
"""
from cachetools import TTLCache

from core.router import Session

# Longer-lived than the per-response TTLCaches (5 min) — a conversation
# session should survive gaps between messages, not just repeated queries.
SESSION_MAXSIZE = 1000
SESSION_TTL_SECONDS = 3600


class SessionStore:

    def __init__(self, maxsize: int = SESSION_MAXSIZE, ttl: int = SESSION_TTL_SECONDS):
        self._sessions: TTLCache[str, Session] = TTLCache(maxsize=maxsize, ttl=ttl)

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session()
        return self._sessions[session_id]
