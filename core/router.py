from agents.chat_agent import ChatAgent
from agents.image_agent import ImageAgent
from agents.rag_agent import RAGAgent
from agents.recommendation_system import RecommendationAgent
from agents.sql_agent import SQLAgent
from agents.weather_agent import WeatherAgent


class Router:

    def __init__(self):
        self.agents = {
            "weather": WeatherAgent(),
            "sql": SQLAgent(),
            "recommendation": RecommendationAgent(),
            "rag": RAGAgent(),
            "image": ImageAgent(),
            "chat": ChatAgent(),
        }

    def route(self, intent, query):
        agent = self.agents.get(intent)

        if agent:
            return agent.handle(query)

        return "Sorry, I couldn't determine the appropriate agent."
