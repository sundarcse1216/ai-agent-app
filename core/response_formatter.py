class ResponseFormatter:
    AGENT_NAMES = {
        "weather": "🌤 Weather Assistant",
        "sql": "🗄 Database Assistant",
        "recommendation": "🎯 Smart Recommender",
        "rag": "📚 Knowledge Assistant",
        "image": "🎨 AI Image Studio",
        "chat": "💬 AI Assistant",
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
