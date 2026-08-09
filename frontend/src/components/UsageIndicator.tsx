interface Props {
  costUsd: number;
  tokens: number;
}

// Small local-model/mini-model costs are often well under $0.001 -- a
// flat toFixed(4) rounds those to "$0.0000", which reads as free even
// though there's a real (just tiny) cost. Show more precision for small
// amounts, less for larger ones.
function formatCost(costUsd: number): string {
  if (costUsd === 0) return "$0.00";
  if (costUsd < 0.001) return `$${costUsd.toFixed(6)}`;
  if (costUsd < 1) return `$${costUsd.toFixed(4)}`;
  return `$${costUsd.toFixed(2)}`;
}

// Running session totals returned on every ChatResponse/StreamResult (see
// core/cost_tracking.py on the backend). Doesn't cover RAG or
// image-generation cost -- see that module's docstring for why.
export function UsageIndicator({ costUsd, tokens }: Props) {
  return (
    <div className="usage-indicator" title="Total LLM usage for this session (excludes RAG and image generation)">
      <span>{tokens.toLocaleString()} tokens</span>
      <span className="usage-indicator__sep">·</span>
      <span>{formatCost(costUsd)}</span>
    </div>
  );
}
