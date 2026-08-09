import type { AgentDirectory, ChatMessage } from "../api/types";
import { AgentBadge } from "./AgentBadge";
import { GeneratedImage } from "./renderers/GeneratedImage";
import { MarkdownResponse } from "./renderers/MarkdownResponse";
import { MonospaceResponse } from "./renderers/MonospaceResponse";

const MARKDOWN_AGENTS = new Set(["chat", "rag"]);

interface Props {
  message: ChatMessage;
  agents: AgentDirectory | null;
}

export function MessageBubble({ message, agents }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`message-bubble ${isUser ? "message-bubble--user" : "message-bubble--assistant"}`}>
      {!isUser && message.agent && (
        <div className="message-bubble__meta">
          <AgentBadge agent={message.agent} agents={agents} />
          {typeof message.elapsed === "number" && (
            <span className="message-bubble__elapsed">{message.elapsed.toFixed(2)}s</span>
          )}
        </div>
      )}

      <div className={`message-bubble__content ${message.isError ? "message-bubble__content--error" : ""}`}>
        {renderContent(message)}
        {message.streaming && <span className="typing-cursor" aria-hidden="true" />}
      </div>
    </div>
  );
}

function renderContent(message: ChatMessage) {
  if (message.role === "user") {
    return <p>{message.text}</p>;
  }

  if (message.agent === "image" && message.imageUrl) {
    return <GeneratedImage imageUrl={message.imageUrl} />;
  }

  if (message.agent && MARKDOWN_AGENTS.has(message.agent)) {
    return <MarkdownResponse text={message.text} />;
  }

  return <MonospaceResponse text={message.text} />;
}
