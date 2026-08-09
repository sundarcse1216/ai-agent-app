import os

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.dependencies import get_router
from config import DATABASE_PATH, EVENTS_DATABASE_PATH, FAISS_INDEX, LLM_PROVIDER, OPENAI_API_KEY
from core.router import Router as AgentRouter

router = APIRouter()


@router.get("/healthz")
def healthz():
    """Liveness: is the process up at all. No dependency checks — this is
    what Docker's HEALTHCHECK / an orchestrator's liveness probe hits."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(agent_router: AgentRouter = Depends(get_router)):
    """Readiness: can this instance actually serve traffic. Checks the
    things that would make every request fail if missing, without doing
    anything as expensive as a full LLM generation on every probe hit."""
    checks = {
        "company_database": os.path.exists(DATABASE_PATH),
        "events_database": os.path.exists(EVENTS_DATABASE_PATH),
        "faiss_index": os.path.exists(FAISS_INDEX),
        "rag_agent_loaded": "rag" in agent_router.agents,
        "llm_provider_configured": _check_llm_provider(),
    }
    ready = all(checks.values())
    return JSONResponse(content={"ready": ready, "checks": checks}, status_code=200 if ready else 503)


def _check_llm_provider() -> bool:
    if LLM_PROVIDER == "ollama":
        try:
            import ollama
            ollama.list()  # cheap: lists local models, no generation call
            return True
        except Exception:
            return False
    if LLM_PROVIDER == "openai":
        # A real API call here would cost money/latency on every readiness
        # probe hit; checking the key is configured is the cheap proxy.
        return bool(OPENAI_API_KEY)
    return False
