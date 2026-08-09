// Used for weather/sql/recommendation responses — these are already
// nicely hand-formatted plain text on the backend (tabulate grids, emoji
// section headers), so a monospace block preserves that formatting
// exactly rather than needing a second structured-data path per agent.
export function MonospaceResponse({ text }: { text: string }) {
  return <pre className="monospace-response">{text}</pre>;
}
