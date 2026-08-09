"""
Offline analysis tool for empirically choosing/verifying
core.semantic_router.CONFIDENCE_THRESHOLD, instead of trusting an assumed
value.

For a labeled set of (query, correct_intent) pairs — reusing every case
from tests/test_cases.py::TestIntentClassifier plus a held-out set of new
queries not used anywhere as exemplars — this computes, per query:

  - correct_score:      similarity to the CORRECT intent's best exemplar
  - best_wrong_score:    similarity to the best-matching WRONG intent
  - margin:              correct_score - best_wrong_score

Two distinct things can go wrong, and the report separates them:

  1. Misrouting (a real accuracy problem, no threshold fixes it): the
     argmax intent is wrong, i.e. margin < 0. This means the exemplar set
     itself needs work for that query shape, not the threshold.
  2. Unnecessary fallback (a tuning problem, not an accuracy problem):
     argmax is right (margin > 0) but correct_score is still below the
     configured threshold, so the query needlessly falls back to an LLM
     call it didn't need.

Run: python scripts/tune_classifier_threshold.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.semantic_router import CONFIDENCE_THRESHOLD, SemanticRouter  # noqa: E402

# Every happy-path case from tests/test_cases.py::TestIntentClassifier,
# plus the two documented negative-case regressions.
FROM_TEST_SUITE = [
    ("What is the weather in Singapore?", "weather"),
    ("Will it rain tomorrow?", "weather"),
    ("What is the temperature today?", "weather"),
    ("Is it cloudy in London?", "weather"),
    ("Check the humidity in Bangkok", "weather"),
    ("Show all employees", "sql"),
    ("What is the salary of John?", "sql"),
    ("Display all departments", "sql"),
    ("Select records from the database", "sql"),
    ("Summarize the contract", "rag"),
    ("What are the kitchen fittings?", "rag"),
    ("Explain the specification document", "rag"),
    ("What does the builder manual say?", "rag"),
    ("Recommend an event in Singapore", "recommendation"),
    ("Suggest a restaurant nearby", "recommendation"),
    ("Any movie suggestions for tonight?", "recommendation"),
    ("Generate an image of a cat", "image"),
    ("Draw a futuristic city", "image"),
    ("Create a logo for my company", "image"),
    ("Make a poster for the event", "image"),
    ("Hello", "chat"),
    ("What is 2 + 2?", "chat"),
    ("Tell me a joke", "chat"),
    ("How are you?", "chat"),
    ("WEATHER IN SINGAPORE", "weather"),
    ("SHOW ALL EMPLOYEES", "sql"),
    ("list the best events", "recommendation"),   # negative case: not sql
    ("explain the weather forecast", "weather"),  # negative case: not rag
]

# Held-out: new phrasings, never used as exemplars or existing test cases.
HELD_OUT = [
    ("Do I need a jacket today in Berlin?", "weather"),
    ("Should I bring an umbrella tomorrow?", "weather"),
    ("What's the UV index like?", "weather"),
    ("Can I take unpaid leave?", "rag"),
    ("What is the flooring material in the spec?", "rag"),
    ("Where should I eat dinner tonight?", "recommendation"),
    ("What's a fun thing to do this evening?", "recommendation"),
    ("Give me an idea for the weekend", "recommendation"),
    ("Sketch a robot for my presentation", "image"),
    ("I need artwork for a birthday card", "image"),
    ("Which department spends the most?", "sql"),
    ("Count how many people work in Marketing", "sql"),
    ("Give me a breakdown of employee salaries", "sql"),
    ("Good morning!", "chat"),
    ("What year is it?", "chat"),
]

LABELED_QUERIES = FROM_TEST_SUITE + HELD_OUT


def main():
    router = SemanticRouter()

    rows = []
    for query, expected in LABELED_QUERIES:
        scores = router.scores(query)
        correct_score = scores[expected]
        best_wrong_intent, best_wrong_score = max(
            ((intent, score) for intent, score in scores.items() if intent != expected),
            key=lambda item: item[1],
        )
        rows.append({
            "query": query,
            "expected": expected,
            "correct_score": correct_score,
            "best_wrong_intent": best_wrong_intent,
            "best_wrong_score": best_wrong_score,
            "margin": correct_score - best_wrong_score,
        })

    misrouted = [r for r in rows if r["margin"] < 0]
    correctly_routed = [r for r in rows if r["margin"] >= 0]

    print(f"Labeled queries: {len(rows)}  (from test suite: {len(FROM_TEST_SUITE)}, held-out: {len(HELD_OUT)})")
    print(f"Configured CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD}")
    print()

    if misrouted:
        print(f"MISROUTED (argmax wrong regardless of threshold) — {len(misrouted)} case(s):")
        for r in misrouted:
            print(
                f"  {r['query']!r}: expected={r['expected']} "
                f"(score={r['correct_score']:.3f}) but best match is "
                f"{r['best_wrong_intent']} (score={r['best_wrong_score']:.3f})"
            )
        print("  -> Fix by adding/adjusting exemplars for the affected intent(s) in "
              "core/intent_exemplars.py, not by changing the threshold.")
        print()
    else:
        print("No misrouted cases — argmax is correct for every labeled query.")
        print()

    unnecessary_fallback = [r for r in correctly_routed if r["correct_score"] < CONFIDENCE_THRESHOLD]
    if unnecessary_fallback:
        print(f"UNNECESSARY FALLBACK (argmax right, but below threshold) — {len(unnecessary_fallback)} case(s):")
        for r in unnecessary_fallback:
            print(f"  {r['query']!r}: correct_score={r['correct_score']:.3f} < threshold {CONFIDENCE_THRESHOLD}")
        print()

    if correctly_routed:
        min_true_positive = min(r["correct_score"] for r in correctly_routed)
        max_near_miss = max(r["best_wrong_score"] for r in rows)
        print(f"min correct-intent score across all correctly-routed queries: {min_true_positive:.3f}")
        print(f"max near-miss (best wrong-intent) score across all queries:   {max_near_miss:.3f}")
        print()
        suggested = round(min_true_positive - 0.03, 2)
        print(
            f"Suggested threshold ceiling: ~{suggested} "
            f"(just under the weakest true positive, so no correctly-routed "
            f"query in this set unnecessarily falls back). Current threshold "
            f"({CONFIDENCE_THRESHOLD}) is "
            f"{'fine as-is' if CONFIDENCE_THRESHOLD <= min_true_positive else 'higher than this — consider lowering it'}."
        )


if __name__ == "__main__":
    main()
