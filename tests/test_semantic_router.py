"""
Unit tests for core/semantic_router.py using a deterministic fake embedding
model (no real sentence-transformers model loaded) — same mocking
philosophy as the rest of the suite (mock the LLM/network/model boundary,
test the logic around it).

Vectors are placed on the unit circle so cosine similarity has a simple
closed form (cos of the angle between two vectors), making the expected
similarity scores easy to reason about and verify by hand.
"""
import math
import unittest
from unittest.mock import patch

import numpy as np


def unit_vector(angle_degrees: float) -> list[float]:
    radians = math.radians(angle_degrees)
    return [math.cos(radians), math.sin(radians)]


class FakeEmbeddingModel:
    """Stands in for SentenceTransformer: looks up a pre-defined vector for
    each known text, so test cases can control similarity scores exactly."""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, texts, normalize_embeddings=True):
        arr = np.array([self.vectors[t] for t in texts], dtype=float)
        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr


def build_router(vectors: dict[str, list[float]], exemplars: dict[str, list[str]]):
    from core.semantic_router import SemanticRouter

    with patch(
        "core.semantic_router._load_embedding_model",
        return_value=FakeEmbeddingModel(vectors),
    ), patch("core.semantic_router.INTENT_EXEMPLARS", exemplars):
        return SemanticRouter()


class TestMaxSimilarityBeatsCentroid(unittest.TestCase):
    """Intent A has one exemplar very close to the query and one far away
    (opposite); intent B has two exemplars moderately close. Max-similarity
    should pick A (its best exemplar is a near-perfect match) even though a
    naive average-of-similarities would favor B (A's average is dragged
    down by its one bad exemplar) — this is exactly the max-vs-centroid
    distinction core/semantic_router.py's docstring describes."""

    def test_max_similarity_wins_over_average(self):
        query = "query"
        vectors = {
            query: unit_vector(0),
            "a1_close": unit_vector(5),      # cos(5°)   ≈ 0.996 to query
            "a2_far": unit_vector(170),      # cos(170°) ≈ -0.985 to query
            "b1_mid": unit_vector(40),       # cos(40°)  ≈ 0.766 to query
            "b2_mid": unit_vector(45),       # cos(45°)  ≈ 0.707 to query
        }
        exemplars = {
            "intent_a": ["a1_close", "a2_far"],
            "intent_b": ["b1_mid", "b2_mid"],
        }
        router = build_router(vectors, exemplars)

        scores = router.scores(query)

        self.assertGreater(scores["intent_a"], scores["intent_b"])
        # Sanity-check the average-similarity story we're contrasting with:
        # intent_a's average would lose to intent_b's despite intent_a
        # having the single best-matching exemplar overall.
        avg_a = (math.cos(math.radians(5)) + math.cos(math.radians(170))) / 2
        avg_b = (math.cos(math.radians(40)) + math.cos(math.radians(45))) / 2
        self.assertLess(avg_a, avg_b)

        intent, _ = router.classify(query)
        self.assertEqual(intent, "intent_a")


class TestConfidenceThreshold(unittest.TestCase):

    def test_above_threshold_returns_intent(self):
        # cos(50°) ≈ 0.643, above CONFIDENCE_THRESHOLD (0.55)
        query = "query"
        vectors = {query: unit_vector(0), "ex": unit_vector(50)}
        router = build_router(vectors, {"only_intent": ["ex"]})

        intent, scores = router.classify(query)

        self.assertEqual(intent, "only_intent")
        self.assertGreaterEqual(scores["only_intent"], 0.55)

    def test_below_threshold_returns_none(self):
        # cos(70°) ≈ 0.342, below CONFIDENCE_THRESHOLD (0.55)
        query = "query"
        vectors = {query: unit_vector(0), "ex": unit_vector(70)}
        router = build_router(vectors, {"only_intent": ["ex"]})

        intent, scores = router.classify(query)

        self.assertIsNone(intent)
        self.assertLess(scores["only_intent"], 0.55)


class TestFallbackTriggering(unittest.TestCase):

    def test_all_intents_below_threshold_triggers_fallback(self):
        query = "query"
        vectors = {
            query: unit_vector(0),
            "weather_ex": unit_vector(80),       # cos(80°) ≈ 0.174
            "sql_ex": unit_vector(100),          # cos(100°) ≈ -0.174
        }
        exemplars = {"weather": ["weather_ex"], "sql": ["sql_ex"]}
        router = build_router(vectors, exemplars)

        intent, scores = router.classify(query)

        self.assertIsNone(intent)
        self.assertTrue(all(s < 0.55 for s in scores.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
