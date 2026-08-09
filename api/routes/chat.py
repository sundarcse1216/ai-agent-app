import json
import os
import re
import time
import uuid

from fastapi import APIRouter, Depends, Header, Response
from fastapi.responses import StreamingResponse

from api.dependencies import get_classifier, get_router, get_session_store
from api.schemas import ChatRequest, ChatResponse
from api.session import SessionStore
from config import OUTPUT_IMAGES
from core.intent_classifier import IntentClassifier
from core.router import Router as AgentRouter
from logger import logger

router = APIRouter()

# ImageAgent's success message always ends with "Saved at:\n{filepath}"
# (agents/image_agent.py) — parsed here at the API boundary rather than
# having ImageAgent return structured data just for this one frontend
# affordance. If that message format ever changes, update this too.
_IMAGE_PATH_PATTERN = re.compile(r"Saved at:\s*\n(.+)", re.MULTILINE)


def _extract_image_url(intent: str, response_text: str) -> str | None:
    if intent != "image":
        return None
    match = _IMAGE_PATH_PATTERN.search(response_text)
    if not match:
        return None
    filename = os.path.basename(match.group(1).strip())
    if os.path.isfile(os.path.join(OUTPUT_IMAGES, filename)):
        return f"/api/images/{filename}"
    return None


def _sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    response: Response,
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    classifier: IntentClassifier = Depends(get_classifier),
    agent_router: AgentRouter = Depends(get_router),
    sessions: SessionStore = Depends(get_session_store),
):
    # Client is expected to generate and persist its own session id (see
    # frontend useSession hook, added in a later phase), but the server
    # tolerates a missing one by minting a fresh one and echoing it back —
    # useful for curl/tooling and for a client's very first request.
    session_id = x_session_id or str(uuid.uuid4())
    response.headers["X-Session-Id"] = session_id
    session = sessions.get_or_create(session_id)

    start = time.time()
    intent = classifier.classify(body.query)
    logger.info(f"[{session_id}] intent={intent} query={body.query!r}")

    agent_response = agent_router.route(intent, body.query, session)
    elapsed = time.time() - start
    is_error = agent_response.startswith("❌")

    logger.info(
        f"API response | session={session_id} agent={intent} is_error={is_error} "
        f"elapsed={elapsed:.3f}s response={agent_response!r} "
        f"session_cost_usd={session.usage.cost_usd:.6f} "
        f"session_tokens={session.usage.prompt_tokens + session.usage.completion_tokens}"
    )

    return ChatResponse(
        agent=intent,
        response=agent_response,
        elapsed=elapsed,
        is_error=is_error,
        image_url=_extract_image_url(intent, agent_response),
        session_cost_usd=session.usage.cost_usd,
        session_tokens=session.usage.prompt_tokens + session.usage.completion_tokens,
    )


@router.post("/chat/stream")
def chat_stream(
    body: ChatRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
    classifier: IntentClassifier = Depends(get_classifier),
    agent_router: AgentRouter = Depends(get_router),
    sessions: SessionStore = Depends(get_session_store),
):
    session_id = x_session_id or str(uuid.uuid4())
    session = sessions.get_or_create(session_id)

    def event_stream():
        start = time.time()
        intent = classifier.classify(body.query)
        logger.info(f"[{session_id}] intent={intent} query={body.query!r} (stream)")
        # Emitted first so the UI can show e.g. "Weather Assistant is
        # thinking" immediately, before any token/result event arrives.
        yield _sse_event("intent", {"agent": intent})

        final_response = None
        for event in agent_router.route_stream(intent, body.query, session):
            if event["type"] == "token":
                yield _sse_event("token", {"content": event["content"]})
            else:
                final_response = event["response"]

        elapsed = time.time() - start
        is_error = final_response.startswith("❌")

        logger.info(
            f"API response | session={session_id} agent={intent} is_error={is_error} "
            f"elapsed={elapsed:.3f}s response={final_response!r} (stream) "
            f"session_cost_usd={session.usage.cost_usd:.6f} "
            f"session_tokens={session.usage.prompt_tokens + session.usage.completion_tokens}"
        )

        yield _sse_event("result", {
            "agent": intent,
            "response": final_response,
            "elapsed": elapsed,
            "is_error": is_error,
            "image_url": _extract_image_url(intent, final_response),
            "session_cost_usd": session.usage.cost_usd,
            "session_tokens": session.usage.prompt_tokens + session.usage.completion_tokens,
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Session-Id": session_id},
    )
