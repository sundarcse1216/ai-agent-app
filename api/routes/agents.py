from fastapi import APIRouter

from core.response_formatter import ResponseFormatter

router = APIRouter()


@router.get("/agents")
def list_agents():
    """Agent display metadata (name + loading copy), so the frontend
    doesn't duplicate ResponseFormatter.AGENT_NAMES / LOADING_MESSAGES
    as a second hand-maintained TypeScript copy."""
    return {
        intent: {
            "name": name,
            "loading_message": ResponseFormatter.LOADING_MESSAGES.get(intent, "Thinking"),
        }
        for intent, name in ResponseFormatter.AGENT_NAMES.items()
    }
