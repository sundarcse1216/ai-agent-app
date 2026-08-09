import { useEffect, useState } from "react";
import { fetchAgents } from "../api/client";
import type { AgentDirectory } from "../api/types";

export function useAgents(): AgentDirectory | null {
  const [agents, setAgents] = useState<AgentDirectory | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchAgents()
      .then((data) => {
        if (!cancelled) setAgents(data);
      })
      .catch((err) => {
        console.error("Failed to load agent directory", err);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return agents;
}
