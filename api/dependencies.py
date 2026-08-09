from fastapi import Request

from api.session import SessionStore
from core.intent_classifier import IntentClassifier
from core.router import Router


def get_classifier(request: Request) -> IntentClassifier:
    return request.app.state.classifier


def get_router(request: Request) -> Router:
    return request.app.state.router


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.sessions
