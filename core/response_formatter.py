class ResponseFormatter:
    AGENT_NAMES = {
        "weather": "🌤 Weather Assistant",
        "sql": "🗄 Database Assistant",
        "recommendation": "🎯 Smart Recommender",
        "rag": "📚 Knowledge Assistant",
        "image": "🎨 AI Image Studio",
        "chat": "💬 AI Assistant",
    }

    LOADING_MESSAGES = {
        "weather": "🌤 Checking weather",
        "sql": "🗄 Querying database",
        "recommendation": "🎯 Finding the best event",
        "rag": "📚 Searching documents",
        "image": "🎨 Creating your image",
        "chat": "💬 Thinking",
    }

    @staticmethod
    def success(agent, response, elapsed):
        title = ResponseFormatter.AGENT_NAMES.get(
            agent.lower(),
            agent.title()
        )
        header = title.center(38)

        return f"""
╔══════════════════════════════════════════════╗
{header}
╚══════════════════════════════════════════════╝

{response}

⏱ Response Time: {elapsed:.2f} seconds
"""

    @staticmethod
    def error(message):
        return f"""
╔══════════════════════════════════════════════╗
                    ❌ Error
╚══════════════════════════════════════════════╝

{message}
"""
