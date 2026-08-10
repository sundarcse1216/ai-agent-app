import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import RequestLoggingMiddleware
from api.routes import agents, chat, health, images
from api.session import SessionStore
from core.intent_classifier import IntentClassifier
from core.router import Router
from database.setup_company_database import setup_company_database
from database.setup_events_database import setup_events_database

# Vite's default dev server origin, always allowed. Production origins
# (e.g. the Netlify site) are added via ALLOWED_ORIGINS, a comma-separated
# env var, so this doesn't need a code change per deploy target.
DEV_FRONTEND_ORIGIN = "http://localhost:5173"
EXTRA_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_company_database()
    setup_events_database()

    # Built once for the process lifetime: IntentClassifier loads the
    # embedding model + exemplar embeddings, Router constructs all six
    # agents (RAGAgent builds/loads its FAISS index here too). Per-session
    # conversation state is handled separately by SessionStore, not by
    # rebuilding these.
    app.state.classifier = IntentClassifier()
    app.state.router = Router()
    app.state.sessions = SessionStore()

    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Smart AI Multi-Agent Assistant API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[DEV_FRONTEND_ORIGIN, *EXTRA_ORIGINS],
        allow_methods=["*"],
        allow_headers=["*"],
        # Custom response headers aren't visible to browser JS unless
        # explicitly exposed — the frontend needs to read X-Session-Id.
        expose_headers=["X-Session-Id"],
    )
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(health.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(agents.router, prefix="/api")
    app.include_router(images.router, prefix="/api")

    return app


app = create_app()
