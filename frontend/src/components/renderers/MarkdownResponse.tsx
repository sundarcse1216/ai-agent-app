import ReactMarkdown from "react-markdown";

// Used for chat/rag responses, which come from an LLM and may contain
// markdown (bold, lists, etc.) worth rendering properly rather than as
// raw text.
export function MarkdownResponse({ text }: { text: string }) {
  return (
    <div className="markdown-response">
      <ReactMarkdown>{text}</ReactMarkdown>
    </div>
  );
}
