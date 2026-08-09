from fastapi import APIRouter

from core.controller import messages as loading_messages
from core.response_formatter import ResponseFormatter

router = APIRouter()


@router.get("/agents")
def list_agents():
    """Agent display metadata (name + loading copy), so the frontend
    doesn't duplicate ResponseFormatter.AGENT_NAMES / controller.messages
    as a second hand-maintained TypeScript copy."""
    return {
        intent: {
            "name": name,
            "loading_message": loading_messages.get(intent, "Thinking"),
        }
        for intent, name in ResponseFormatter.AGENT_NAMES.items()
    }
