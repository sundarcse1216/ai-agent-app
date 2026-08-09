"""
Embedding-based intent router.

Computes cosine similarity between an incoming query and a fixed set of
per-intent exemplar phrases (core/intent_exemplars.py), and returns the
intent whose *closest single exemplar* is most similar — nearest-neighbor,
not a blurred per-intent centroid. A centroid (mean of all of an intent's
exemplar vectors) would sit in a semantic no-man's-land for intents whose
exemplars are phrasally diverse (e.g. "rag" spans both HR-policy questions
and construction-spec questions — averaging those vectors doesn't represent
either well). Max-similarity-to-any-exemplar is also easier to debug: you
can always point to *which* exemplar triggered a match.

Below CONFIDENCE_THRESHOLD, classify() returns None so the caller
(IntentClassifier) can fall back to an LLM call for the ambiguous case.
"""
import functools

import numpy as np
from sentence_transformers import SentenceTransformer

from core.intent_exemplars import INTENT_EXEMPLARS

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Initial value, chosen from bge-small's typical similarity ranges for
# related vs. unrelated short sentences. See
# scripts/tune_classifier_threshold.py for how to empirically verify/adjust
# this against a held-out query set rather than trusting it blindly.
CONFIDENCE_THRESHOLD = 0.55


@functools.lru_cache(maxsize=1)
def _load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    # Cached so repeated SemanticRouter() construction (e.g. across tests)
    # reuses the same loaded model weights instead of reloading them from
    # disk every time. In the running app there's only ever one instance
    # anyway (built once at Router startup, same lifecycle as RAGAgent's
    # FAISS index).
    return SentenceTransformer(model_name)


class SemanticRouter:

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model = _load_embedding_model(model_name)
        # {intent: (n_exemplars, dim) L2-normalized embedding matrix}, built
        # once at construction so cosine similarity is a plain dot product.
        self._exemplar_embeddings: dict[str, np.ndarray] = {
            intent: self._embed(phrases)
            for intent, phrases in INTENT_EXEMPLARS.items()
        }

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))

    def scores(self, query: str) -> dict[str, float]:
        """Max cosine similarity between `query` and each intent's exemplars."""
        query_embedding = self._embed([query])[0]
        return {
            intent: float(np.max(exemplar_matrix @ query_embedding))
            for intent, exemplar_matrix in self._exemplar_embeddings.items()
        }

    def classify(self, query: str) -> tuple[str | None, dict[str, float]]:
        """Returns (best_intent, all_scores). best_intent is None if the top
        score is below CONFIDENCE_THRESHOLD (caller should fall back)."""
        intent_scores = self.scores(query)
        best_intent = max(intent_scores, key=intent_scores.get)
        if intent_scores[best_intent] < CONFIDENCE_THRESHOLD:
            return None, intent_scores
        return best_intent, intent_scores
