interface Props {
  label: string;
}

// Console-app equivalent: ui.py's Spinner. This is the web version — a
// small animated dots indicator paired with per-agent loading copy
// (sourced from GET /api/agents, same text the console app uses).
export function LoadingIndicator({ label }: Props) {
  return (
    <div className="loading-indicator">
      <span className="loading-indicator__dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <span>{label}</span>
    </div>
  );
}
