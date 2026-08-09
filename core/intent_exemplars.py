"""
Exemplar phrases for the semantic intent router (core/semantic_router.py).

Unlike the old INTENT_KEYWORDS (single words/short phrases, tuned to avoid
substring collisions — e.g. "list" and "explain" had to be removed because
they accidentally matched the wrong intent), these are full, realistic
sentences. Embeddings compare meaning, not substrings, so there's no
collision problem to design around — the more natural and varied the
phrasing, the better the router generalizes.

Seeded from demo.py's curated per-agent queries plus the coverage the old
INTENT_KEYWORDS implied, then expanded with paraphrases.
"""

INTENT_EXEMPLARS: dict[str, list[str]] = {
    "weather": [
        "What is the weather in Singapore?",
        "How is the weather in Tokyo right now?",
        "Is it going to rain in London today?",
        "What's the temperature outside?",
        "Will it be sunny tomorrow in Paris?",
        "Check the humidity in Bangkok",
        "Is it cloudy in New York?",
        "What's the forecast for this weekend?",
        "How windy is it in Chicago?",
        "Do I need an umbrella today?",
    ],
    "rag": [
        "Summarize the contract for me",
        "What are the kitchen fittings in the specification?",
        "Explain the specification document",
        "What does the builder manual say about the bathroom?",
        "How many days of annual leave do employees get?",
        "What dental benefits are available?",
        "What is the company's work-from-home policy?",
        "What happens if an employee resigns?",
        "What is the notice period for resignation?",
        "Tell me about the maternity leave policy",
        "What electrical fittings are provided?",
        "What is the company's dress code?",
        "How does the grievance process work?",
        "What is covered under employee insurance?",
    ],
    "recommendation": [
        "Recommend an event in Singapore",
        "Suggest a restaurant nearby",
        "Any movie suggestions for tonight?",
        "What should I do this weekend in Tokyo?",
        "Recommend an outdoor activity for tomorrow",
        "What indoor events are available today?",
        "Suggest something fun to do in the city",
        "What's a good event to attend this evening?",
        "Recommend a place to visit given the weather",
        "Give me an idea for something to do this weekend",
        "I need suggestions for activities",
    ],
    "image": [
        "Generate an image of a cat",
        "Draw a futuristic city",
        "Create a logo for my company",
        "Make a poster for the event",
        "Draw a watercolor painting of a mountain lake",
        "Generate a cyberpunk city at night",
        "Create an illustration of a dragon",
        "Design a banner for our sale",
        "Make a flyer for the concert",
        "Paint a picture of a sunset",
        "Generate an image for the Singapore event",
        "Create a poster for the birthday celebration",
        "Design a banner for the city festival",
        "Can you generate an image to celebrate our anniversary?",
    ],
    "sql": [
        "Show all employees",
        "What is the salary of John?",
        "Display all departments",
        "Select records from the database",
        "List all employees",
        "Show the salary of every employee in Engineering",
        "What is the total budget across all departments?",
        "Who earns the highest salary?",
        "How many employees are in the Marketing department?",
        "Show me the department with the largest budget",
    ],
    "chat": [
        "Hello! What can you help me with?",
        "What is the capital of France?",
        "Can you explain what machine learning is in simple terms?",
        "Tell me a joke",
        "How are you?",
        "What's 2 plus 2?",
        "Who won the world cup in 2018?",
        "Can you help me brainstorm some ideas?",
        "Thanks, that's helpful",
        "What can you do?",
        "What year is it?",
        "What day of the week is it?",
        "Hi, my name is Alex and I'm from Chicago",
        "I'm Maria, I live in Barcelona",
        "Nice to meet you, I'm visiting from Brazil",
        "Hello, this is Raj calling from Mumbai",
        "I'm from Australia",
    ],
}
