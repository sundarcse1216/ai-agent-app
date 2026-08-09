import time

from starlette.middleware.base import BaseHTTPMiddleware

from logger import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """HTTP-layer counterpart to Controller.process's existing
    query/timing logging — same logger, extended to cover the transport
    layer (method, path, session id, status, latency) as well as the
    business layer (intent, query), which api/routes/chat.py already logs."""

    async def dispatch(self, request, call_next):
        start = time.time()
        session_id = request.headers.get("X-Session-Id", "-")

        response = await call_next(request)

        elapsed = time.time() - start
        logger.info(
            f"{request.method} {request.url.path} session={session_id} "
            f"status={response.status_code} latency={elapsed:.3f}s"
        )
        return response
