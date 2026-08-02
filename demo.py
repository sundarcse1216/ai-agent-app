"""
Demo script — Smart AI Multi-Agent Assistant
Runs a curated set of queries through every agent to showcase capabilities.

Usage:
    python demo.py
    python demo.py --agent weather        # run a single agent only
    python demo.py --agent sql
    python demo.py --agent rag
    python demo.py --agent recommendation
    python demo.py --agent chat
    python demo.py --agent image
"""
import argparse
import sys
from datetime import date, timedelta

from core.controller import Controller
from database.setup_company_database import setup_company_database
from database.setup_events_database import setup_events_database

today = date.today().isoformat()
tomorrow = (date.today() + timedelta(days=1)).isoformat()
day_after = (date.today() + timedelta(days=2)).isoformat()

# ---------------------------------------------------------------------------
# Demo queries grouped by agent
# ---------------------------------------------------------------------------
DEMO_SECTIONS = [
    {
        "label": "weather",
        "title": "🌤  Weather Agent",
        "queries": [
            "What is the weather in Singapore?",
            "How is the weather in Tokyo right now?",
            "Is it going to rain in London today?",
        ],
    },
    {
        "label": "sql",
        "title": "🗄  Database Agent",
        "queries": [
            "List all employees",
            "Show the salary of every employee in Engineering",
            "What is the total budget across all departments?",
            "Who earns the highest salary?",
        ],
    },
    {
        "label": "recommendation",
        "title": "🎯  Recommendation Agent",
        "queries": [
            f"Recommend an event in Singapore on {today}",
            f"Suggest an outdoor activity for {tomorrow}",
            f"What indoor events are available on {day_after}?",
        ],
    },
    {
        "label": "rag",
        "title": "📚  Knowledge Agent (RAG)",
        "queries": [
            "How many days of annual leave do employees get?",
            "What dental benefits are available?",
            "What is the company's work-from-home policy?",
            "What are the kitchen fittings provided in the specification?",
            "What happens if an employee resigns?",
        ],
    },
    {
        "label": "chat",
        "title": "💬  Chat Agent",
        "queries": [
            "Hello! What can you help me with?",
            "What is the capital of France?",
            "Can you explain what machine learning is in simple terms?",
        ],
    },
    {
        "label": "image",
        "title": "🎨  Image Agent",
        "queries": [
            "Generate a cyberpunk city at night",
            "Draw a watercolor painting of a mountain lake",
        ],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def divider(char="═", width=62):
    print(char * width)


def section_header(title):
    print()
    divider()
    print(f"  {title}")
    divider()


def run_query(controller, query):
    print(f"\n  Query : {query}")
    print(f"  {'─' * 56}")
    try:
        response = controller.process(query)
        print(response)
    except KeyboardInterrupt:
        print("\n  [Interrupted]")
        sys.exit(0)
    except Exception as e:
        print(f"  [Skipped — {e}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Demo for Smart AI Multi-Agent Assistant")
    parser.add_argument(
        "--agent",
        choices=[s["label"] for s in DEMO_SECTIONS],
        help="Run only a specific agent's demo queries",
    )
    args = parser.parse_args()

    divider("═")
    print("      Smart AI Multi-Agent Assistant — Demo Run")
    divider("═")

    print("\nInitialising databases ...")
    setup_company_database()
    setup_events_database()
    print("Ready.\n")

    controller = Controller()

    sections = (
        [s for s in DEMO_SECTIONS if s["label"] == args.agent]
        if args.agent
        else DEMO_SECTIONS
    )

    for section in sections:
        section_header(section["title"])
        for query in section["queries"]:
            run_query(controller, query)

    print()
    divider()
    print("  Demo complete.")
    divider()


if __name__ == "__main__":
    main()
