"""
Configuration File
"""

import os

from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# OpenWeather API
# -----------------------------
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# -----------------------------
# Open AI
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# -----------------------------
# Ollama Model
# -----------------------------
OLLAMA_MODEL = "llama3.2"

# -----------------------------
# LLM Provider
# -----------------------------
# Which LLMProvider backs LLMService/ChatAgent/RAGAgent/the classifier's LLM
# fallback: "ollama" (local, default) or "openai" (cloud).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# -----------------------------
# LangSmith tracing (optional)
# -----------------------------
# Not read into named variables here — the langsmith SDK reads
# LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY / LANGCHAIN_PROJECT directly from
# the environment, and load_dotenv() above already puts .env into os.environ.
# Tracing is a no-op (near-zero overhead) unless LANGCHAIN_TRACING_V2=true.
# RAGAgent's LangChain chain gets traced automatically. Every other agent's
# non-streaming LLMProvider.chat_messages calls (weather/sql/recommendation/
# classifier-fallback/non-streaming chat) are @traceable too — but NOT the
# streaming path (core/providers/*.py's stream_messages), since tracing that
# generator hung requests when nested through this app's streaming call
# chain (see the comment on stream_messages in each provider).

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAISS_INDEX = os.path.join(BASE_DIR, "vectorstore", "faiss_index")

DOCUMENT_PATH = os.path.join(BASE_DIR, "documents")

OUTPUT_IMAGES = os.path.join(BASE_DIR, "outputs", "generated_images")

LOG_PATH = os.path.join(BASE_DIR, "outputs", "logs", "app.log")

DATABASE_PATH = os.path.join(BASE_DIR, "database", "company.db")
EVENTS_DATABASE_PATH = os.path.join(BASE_DIR, "database", "events.db")

MEMORY_WINDOW = 50

# -----------------------------
# Per-session cost budget
# -----------------------------
# Hard cap on tracked LLM spend per session (see core/cost_tracking.py).
# Always $0 for Ollama calls, so this only actually limits anything when
# LLM_PROVIDER=openai. Does not cover RAGAgent or image generation costs
# (see core/cost_tracking.py's module docstring for why).
MAX_COST_PER_SESSION_USD = float(os.getenv("MAX_COST_PER_SESSION_USD", "0.00010"))
