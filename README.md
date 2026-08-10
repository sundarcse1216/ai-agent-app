# Smart AI Multi-Agent Assistant

A multi-agent AI assistant: a hybrid semantic classifier routes each query to one of six agents (weather, database/SQL, event recommendations, HR-and-spec document search via RAG, image generation, general chat), each of which can talk to a local (Ollama) or cloud (OpenAI) LLM depending on configuration. It runs three ways from the same codebase — a console REPL, a FastAPI web backend, and a React frontend on top of that backend — and streams responses over Server-Sent Events.

## Architecture

```
┌──────────────┐        ┌───────────────────┐        ┌───────────────────────┐
│   Frontend    │  HTTP  │   FastAPI backend  │        │   core/ + agents/     │
│  (React/Vite) │◄──────►│      (api/)        │◄──────►│  (shared business     │
└──────────────┘  SSE   └───────────────────┘        │   logic, also used    │
                                                        │   by main.py/demo.py) │
                                                        └───────────────────────┘
                                                                    │
                                              ┌─────────────────────┼─────────────────────┐
                                              ▼                     ▼                     ▼
                                        Ollama / OpenAI      SQLite (company/       FAISS index
                                        (LLMProvider)         events.db)          (documents/)
```

- **`main.py`** — console REPL. **`demo.py`** — scripted showcase of all six agents. **`api/`** — the FastAPI web layer. All three sit on top of the same `core/`/`agents/`/`database/` business logic — nothing is duplicated between the console and web paths.
- **`core/intent_classifier.py`** — hybrid classifier: a fast embedding-based semantic router (`core/semantic_router.py`, no LLM call) handles the common case; below its confidence threshold, an LLM call resolves the ambiguous query.
- **`core/llm_provider.py`** / **`core/providers/`** — swappable LLM backend (Ollama or OpenAI), selected via `LLM_PROVIDER`. Most agents get this for free through `core/llm_service.py`; `RAGAgent`'s LangChain retrieval chain selects its LangChain chat-model class the same way, one layer down.
- **`core/router.py`** — routes an intent to its agent. `Session` (`chat_history`, `rag_history`) carries the two stateful agents' conversation memory — agents themselves are stateless and pooled (built once), so one shared instance can safely serve many concurrent sessions.
- **`api/session.py`** — per-`X-Session-Id` `Session` store, so concurrent browser sessions on the shared backend process don't bleed conversation memory into each other.

## Running

### Console

```bash
pip install -r requirements.txt      # or requirements-dev.txt to also get pytest
python main.py                        # interactive REPL ("exit" to quit)
python demo.py                        # runs curated queries through every agent
python demo.py --agent sql            # one agent's demo queries only
```

### Web (backend + frontend, without Docker)

```bash
cp .env.example .env                  # fill in the keys you need, see below
pip install -r requirements.txt
uvicorn api.app:app --reload --port 8000

# in a second terminal
cd frontend
npm install
npm run dev                           # http://localhost:5173, proxies /api to :8000
```

### Docker

```bash
cp .env.example .env
docker compose up --build
# add a local Ollama container too, if LLM_PROVIDER=ollama and you don't
# already have Ollama running on the host:
docker compose --profile local-llm up --build
```

Frontend on `http://localhost:5173`, backend on `http://localhost:8000`. `database/`, `vectorstore/`, and `outputs/` are volume-mounted so SQLite data, the FAISS index, and logs persist across rebuilds.

## Configuration

All read from `.env` (see `.env.example`):

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `ollama` (default, local) or `openai` (cloud) — backs chat/RAG/classification-fallback/weather-location-extraction/SQL-generation |
| `OPENAI_API_KEY` | Required if `LLM_PROVIDER=openai`, and always required for image generation (uses OpenAI's `gpt-image-1` regardless of `LLM_PROVIDER`) |
| `OPENAI_CHAT_MODEL` | Override the OpenAI chat model (default `gpt-4o-mini`) |
| `WEATHER_API_KEY` | weatherapi.com, used by the weather agent and the recommendation agent |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | Optional [LangSmith](https://smith.langchain.com) tracing, no-op unless `LANGCHAIN_TRACING_V2=true` |

## Tests

```bash
python -m pytest tests/ -v
```

Mocks the LLM and weather-API boundaries — no Ollama/network access needed to run. `scripts/tune_classifier_threshold.py` is a separate, standalone tool for empirically checking the semantic router's confidence threshold against a labeled query set (not part of the automated suite).

## Why these choices

A few decisions worth knowing the reasoning behind, since they're not the only reasonable option:

- **SSE, not WebSocket, for streaming** (`POST /api/chat/stream`). Traffic is one request in, a stream of tokens out — WebSocket's bidirectional channel buys nothing here and costs more to build/operate (connection lifecycle, keep-alives). Consumed on the frontend via a hand-rolled `fetch()` + `ReadableStream` reader rather than the browser `EventSource` API, since `EventSource` only supports GET and this needs a POST body.
- **Session id via `X-Session-Id` header, not a cookie.** Client-generated UUID in `localStorage`. Fully visible in frontend code — no `SameSite`/credentialed-CORS configuration needed at this no-auth scope.
- **Agents are stateless; `Session` carries conversation history.** `ChatAgent`/`RAGAgent` take history as a parameter and return the updated history, rather than storing it on `self`. This is what makes it safe for one pooled agent instance (built once at startup, since building it is expensive) to serve many concurrent sessions without their conversations bleeding into each other.
- **In-memory session store, deliberately.** `api/session.py`'s `TTLCache` is a single-process solution — correct and sufficient for this app's scope (no auth, not designed for horizontal scaling). A multi-instance deployment would need this moved to something shared like Redis.
- **`is_error` derived from the existing `"❌"` message prefix, not a rewrite to structured exceptions.** Every agent already returns friendly in-band error strings for *expected* failure modes (bad input, missing API key, no results). The API boundary translates that convention into a `bool` for the frontend rather than restructuring six agents. Genuinely unexpected failures still surface as real HTTP error codes.

## Project layout

```
agents/       one BaseAgent subclass per capability (weather, sql, recommendation, rag, image, chat)
core/         intent classification, routing, LLM provider abstraction, response formatting
api/          FastAPI app: routes, session store, request middleware
database/     SQLite setup/seed scripts + a thin events repository
frontend/     React + TypeScript + Vite SPA
scripts/      standalone dev tools (classifier threshold tuning)
tests/        unittest-based suite (also pytest-compatible)
```
