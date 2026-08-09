interface Props {
  message: string;
  onDismiss: () => void;
}

// For transport/infrastructure failures (network error, non-2xx from the
// API) — distinct from an agent's own in-band "❌ ..." messages, which
// render as a normal (styled-as-error) assistant message bubble instead.
export function ErrorBanner({ message, onDismiss }: Props) {
  return (
    <div className="error-banner">
      <span>{message}</span>
      <button type="button" onClick={onDismiss} aria-label="Dismiss">
        ×
      </button>
    </div>
  );
}
