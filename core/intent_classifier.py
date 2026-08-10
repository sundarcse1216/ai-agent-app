import json

from core.llm_provider_factory import get_llm_provider
from core.semantic_router import SemanticRouter
from logger import logger

VALID_INTENTS = ("weather", "rag", "recommendation", "image", "sql", "chat")

_FALLBACK_SYSTEM_PROMPT = """
You are an intent classification assistant for a multi-agent system.

Classify the user's query into exactly one of these intents:
- weather: questions about current weather, temperature, forecast, rain, etc.
- rag: questions about company HR policies/benefits/handbook, or the building specification document
- recommendation: requests for event, activity, or restaurant recommendations
- image: requests to generate, draw, create, or design an image
- sql: questions about employee or department records in the company database
- chat: general conversation, greetings, or anything that doesn't fit the above

Return ONLY valid JSON in this exact format:
{"intent": "<one of: weather, rag, recommendation, image, sql, chat>"}

Rules:
- Return only JSON.
- Never explain.
- If genuinely unsure, use "chat".
"""


class IntentClassifier:
    """Hybrid classifier: a fast embedding-based semantic router
    (core/semantic_router.py) handles the common case with no LLM call;
    below its confidence threshold, an LLM call resolves the ambiguous
    query. Replaces the old pure-keyword-substring matcher."""

    def __init__(self):
        self._router = SemanticRouter()

    def classify(self, query: str) -> str:
        if not query.strip():
            return "chat"

        intent, scores = self._router.classify(query)
        logger.info(f"Semantic router confidence threshold for {query!r}: {intent}: {scores}")
        if intent is not None:
            return intent

        return self._classify_with_llm(query)

    @staticmethod
    def _classify_with_llm(query: str) -> str:
        logger.info("Fallback to LLM Classifier")
        try:
            response = get_llm_provider().chat(
                system_prompt=_FALLBACK_SYSTEM_PROMPT,
                user_prompt=query,
                json_mode=True,
            )
            response = response.replace("```json", "").replace("```", "").strip()
            intent = json.loads(response).get("intent")
            if intent in VALID_INTENTS:
                return intent
            logger.error(f"LLM classifier fallback returned invalid intent: {intent!r}")
        except Exception:
            logger.exception("LLM classifier fallback failed")

        return "chat"
