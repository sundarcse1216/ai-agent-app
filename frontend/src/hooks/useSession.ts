import { useState } from "react";

const STORAGE_KEY = "ai-agent-app.session-id";

// Client-generated UUID persisted in localStorage, sent as X-Session-Id on
// every request — this is what keeps chat/RAG conversation memory scoped
// to this browser tab/profile instead of bleeding across users on the
// shared backend process (see core/router.Session, api/session.py).
function getOrCreateSessionId(): string {
  const existing = localStorage.getItem(STORAGE_KEY);
  if (existing) {
    return existing;
  }
  const generated = crypto.randomUUID();
  localStorage.setItem(STORAGE_KEY, generated);
  return generated;
}

export function useSession(): string {
  const [sessionId] = useState(getOrCreateSessionId);
  return sessionId;
}
