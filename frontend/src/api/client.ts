import type { AgentDirectory, ChatResponse } from "./types";

// Set VITE_API_BASE_URL for a non-default backend (e.g. in Docker); empty
// string means "same origin, relative paths" for local dev via Vite's proxy.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export async function fetchAgents(): Promise<AgentDirectory> {
  const res = await fetch(`${API_BASE_URL}/api/agents`);
  if (!res.ok) {
    throw new Error(`Failed to load agent directory: ${res.status}`);
  }
  return res.json();
}

export async function sendChat(query: string, sessionId: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": sessionId,
    },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    throw new Error(`Chat request failed: ${res.status}`);
  }
  return res.json();
}

export function imageUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export { API_BASE_URL };
