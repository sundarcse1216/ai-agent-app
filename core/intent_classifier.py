class IntentClassifier:

    INTENT_KEYWORDS = {
        "weather": [
            "weather", "temperature", "forecast", "rain",
            "sunny", "cloudy", "humidity", "wind", "climate"
        ],

        "rag": [
            # document-type triggers
            "document", "pdf", "summarize", "summary",
            "specification", "manual", "report", "contract",
            "builder", "kitchen", "bedroom", "bathroom",
            "electrical", "fittings", "provided",
            # HR / policy triggers (matches handbook, leave policy, benefits guide)
            "resign", "resignation", "notice period",
            "leave", "annual leave", "sick leave", "maternity", "paternity",
            "benefits", "dental", "insurance", "medical",
            "policy", "policies", "handbook",
            "probation", "allowance", "conduct",
            "grievance", "dress code", "offboard",
        ],

        "recommendation": [
            "recommend", "recommendation", "suggest", "suggestion",
            "movie", "event", "restaurant"
        ],

        "image": [
            "image", "draw", "generate image", "picture",
            "photo", "illustration", "logo",
            "poster", "design", "banner", "flyer"
        ],

        "sql": [
            "database", "employee", "employees", "sql",
            "salary", "table", "tables", "record",
            "records", "department", "departments",
            "select", "show", "display"
        ],
    }

    def classify(self, query: str):

        text = query.lower()

        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return intent

        return "chat"