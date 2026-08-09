import type { AgentDirectory, AgentIntent } from "../api/types";

interface Props {
  agent: AgentIntent;
  agents: AgentDirectory | null;
}

export function AgentBadge({ agent, agents }: Props) {
  const label = agents?.[agent]?.name ?? agent;
  return <span className="agent-badge">{label}</span>;
}
