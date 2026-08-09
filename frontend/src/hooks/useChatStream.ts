import { useCallback, useState } from "react";
import { streamChat } from "../api/chatStream";
import type { StreamResult } from "../api/chatStream";
import type { AgentIntent } from "../api/types";

type StreamStatus = "idle" | "streaming" | "done" | "error";

interface StreamState {
  status: StreamStatus;
  agentIntent: AgentIntent | null;
  tokens: string;
  result: StreamResult | null;
  error: string | null;
}

const IDLE_STATE: StreamState = {
  status: "idle",
  agentIntent: null,
  tokens: "",
  result: null,
  error: null,
};

export function useChatStream(sessionId: string) {
  const [state, setState] = useState<StreamState>(IDLE_STATE);

  const send = useCallback(
    async (query: string) => {
      setState({ status: "streaming", agentIntent: null, tokens: "", result: null, error: null });
      try {
        await streamChat(query, sessionId, {
          onIntent: (agent) => setState((s) => ({ ...s, agentIntent: agent })),
          onToken: (content) => setState((s) => ({ ...s, tokens: s.tokens + content })),
          onResult: (result) => setState((s) => ({ ...s, status: "done", result })),
        });
      } catch (err) {
        setState((s) => ({
          ...s,
          status: "error",
          error: err instanceof Error ? err.message : "Streaming failed.",
        }));
      }
    },
    [sessionId]
  );

  const reset = useCallback(() => setState(IDLE_STATE), []);

  return { ...state, send, reset };
}
