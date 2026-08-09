// Mirrors api/schemas.py (backend) and api/routes/agents.py's response shape.

export type AgentIntent = "weather" | "sql" | "recommendation" | "rag" | "image" | "chat";

export interface ChatResponse {
  agent: AgentIntent;
  response: string;
  elapsed: number;
  is_error: boolean;
  image_url: string | null;
  // Running totals for this session (see core/cost_tracking.py on the
  // backend). Doesn't cover RAG or image-generation cost.
  session_cost_usd: number;
  session_tokens: number;
}

export interface AgentMeta {
  name: string;
  loading_message: string;
}

export type AgentDirectory = Record<AgentIntent, AgentMeta>;

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  agent?: AgentIntent;
  isError?: boolean;
  imageUrl?: string | null;
  elapsed?: number;
  streaming?: boolean;
}
