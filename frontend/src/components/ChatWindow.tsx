import { useEffect, useRef, useState } from "react";
import type { AgentDirectory, ChatMessage } from "../api/types";
import { useChatStream } from "../hooks/useChatStream";
import { useSession } from "../hooks/useSession";
import { ErrorBanner } from "./ErrorBanner";
import { InputBar } from "./InputBar";
import { LoadingIndicator } from "./LoadingIndicator";
import { MessageBubble } from "./MessageBubble";
import { UsageIndicator } from "./UsageIndicator";

interface Props {
  agents: AgentDirectory | null;
}

interface Usage {
  costUsd: number;
  tokens: number;
}

export function ChatWindow({ agents }: Props) {
  const sessionId = useSession();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [usage, setUsage] = useState<Usage | null>(null);
  const stream = useChatStream(sessionId);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, stream.tokens, stream.status]);

  // Commit the finished streamed turn into the message list, then reset
  // the stream hook so it's ready for the next send().
  useEffect(() => {
    if (stream.status === "done" && stream.result) {
      const result = stream.result;
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: result.response,
          agent: result.agent,
          isError: result.is_error,
          imageUrl: result.image_url,
          elapsed: result.elapsed,
        },
      ]);
      setUsage({ costUsd: result.session_cost_usd, tokens: result.session_tokens });
      stream.reset();
    }
  }, [stream.status, stream.result]); // eslint-disable-line react-hooks/exhaustive-deps

  function handleSend(query: string) {
    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", text: query }]);
    stream.send(query);
  }

  const isPending = stream.status === "streaming";
  const loadingLabel = stream.agentIntent
    ? (agents?.[stream.agentIntent]?.loading_message ?? "Thinking...")
    : "Thinking...";

  return (
    <div className="chat-window">
      {stream.status === "error" && stream.error && (
        <ErrorBanner message={stream.error} onDismiss={stream.reset} />
      )}

      <div className="chat-window__messages">
        {messages.length === 0 && !isPending && (
          <div className="chat-window__empty">
            <p>
              Ask about the weather, company records, event recommendations, HR
              policies, or generate an image.
            </p>
          </div>
        )}

        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} agents={agents} />
        ))}

        {isPending && stream.tokens && stream.agentIntent && (
          <MessageBubble
            message={{
              id: "in-progress",
              role: "assistant",
              text: stream.tokens,
              agent: stream.agentIntent,
              streaming: true,
            }}
            agents={agents}
          />
        )}
        {isPending && !stream.tokens && <LoadingIndicator label={loadingLabel} />}

        <div ref={bottomRef} />
      </div>

      {usage && <UsageIndicator costUsd={usage.costUsd} tokens={usage.tokens} />}
      <InputBar onSend={handleSend} disabled={isPending} />
    </div>
  );
}
