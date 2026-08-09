import { API_BASE_URL } from "./client";
import type { AgentIntent } from "./types";

export interface StreamResult {
  agent: AgentIntent;
  response: string;
  elapsed: number;
  is_error: boolean;
  image_url: string | null;
  session_cost_usd: number;
  session_tokens: number;
}

export interface StreamHandlers {
  onIntent?: (agent: AgentIntent) => void;
  onToken?: (content: string) => void;
  onResult?: (result: StreamResult) => void;
}

// Hand-rolled SSE-over-POST client: browser EventSource only supports GET,
// so POST /api/chat/stream is consumed via fetch() + ReadableStream +
// TextDecoder instead. No new dependency — this app only needs a single
// forward pass through one response, no reconnect/retry semantics.
export async function streamChat(
  query: string,
  sessionId: string,
  handlers: StreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": sessionId,
    },
    body: JSON.stringify({ query }),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`Streaming chat request failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let separatorIndex: number;
    while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, separatorIndex);
      buffer = buffer.slice(separatorIndex + 2);
      dispatchFrame(frame, handlers);
    }
  }
}

function dispatchFrame(frame: string, handlers: StreamHandlers) {
  let eventType = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) {
      eventType = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      data = line.slice("data:".length).trim();
    }
  }
  if (!data) return;

  const parsed = JSON.parse(data);
  if (eventType === "intent") {
    handlers.onIntent?.(parsed.agent);
  } else if (eventType === "token") {
    handlers.onToken?.(parsed.content);
  } else if (eventType === "result") {
    handlers.onResult?.(parsed);
  }
}
