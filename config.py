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
# Stability AI
# -----------------------------
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
IMAGE_MODEL = "stable-diffusion-xl-1024-v1-0"

# -----------------------------
# Open AI
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Ollama Model
# -----------------------------
OLLAMA_MODEL = "llama3.2"

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

MEMORY_WINDOW = 5
