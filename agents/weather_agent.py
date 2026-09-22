import requests
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from config import LLM_PROVIDER, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_CHAT_MODEL, WEATHER_API_KEY
from core.cache import make_cache
from core.conversation_context import format_recent_turns
from core.cost_tracking import calculate_openai_cost
from logger import logger

emoji_map = {
    "Sunny": "☀️",
    "Clear": "🌙",
    "Partly cloudy": "⛅",
    "Cloudy": "☁️",
    "Overcast": "☁️",
    "Mist": "🌫️",
    "Fog": "🌫️",
    "Patchy rain nearby": "🌦️",
    "Light rain": "🌦️",
    "Moderate rain": "🌧️",
    "Heavy rain": "🌧️",
    "Thunderstorm": "⛈️",
    "Snow": "❄️"
}
weather_cache = make_cache()


class LocationExtraction(BaseModel):
    """Schema for the one LLM call this agent makes — everything after this
    (get_weather()'s real API call, format_weather()'s deterministic emoji
    card) has no LLM involved and is untouched by this."""
    location: str | None = Field(
        description=(
            "A city, region, or country the user is asking about "
            "(weatherapi.com accepts any of these, not just cities). "
            "Null if no location is named anywhere, including earlier turns."
        )
    )


_EXTRACTION_SYSTEM_PROMPT = """
You are an information extraction assistant.

Extract the location the user is asking about — a city, region, or
country. If the latest message doesn't name one but an earlier part of
this conversation did (including something as indirect as "I'm from X"),
use that instead. If no location is found anywhere, return null.
"""


class _UsageCallback(BaseCallbackHandler):
    """Same approach as agents/chat_agent.py's _UsageCallback (duplicated
    rather than shared — each agent file in this codebase is self-contained).
    Needed because with_structured_output() calls the model directly via
    LangChain, bypassing LLMProvider's ambient-contextvar cost tracking the
    old LLMService.chat() call here used to get for free."""

    def __init__(self, usage, model: str, is_openai: bool):
        self.usage = usage
        self.model = model
        self.is_openai = is_openai

    def on_llm_end(self, response, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                message = getattr(gen, "message", None)
                usage_metadata = getattr(message, "usage_metadata", None) if message else None
                if not usage_metadata:
                    continue
                prompt_tokens = usage_metadata.get("input_tokens", 0)
                completion_tokens = usage_metadata.get("output_tokens", 0)
                cost = (
                    calculate_openai_cost(self.model, prompt_tokens, completion_tokens)
                    if self.is_openai else 0.0
                )
                self.usage.add(prompt_tokens, completion_tokens, cost)


def _usage_config(usage) -> dict:
    if usage is None:
        return {}
    return {"callbacks": [_UsageCallback(usage, OPENAI_CHAT_MODEL, LLM_PROVIDER == "openai")]}


def build_chat_model():
    """Same provider-selection idea as agents/rag_agent.py::build_chat_model —
    LangChain needs its own chat-model class, it can't use this project's
    LLMProvider interface directly."""
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(model=OPENAI_CHAT_MODEL, api_key=OPENAI_API_KEY)
    return ChatOllama(model=OLLAMA_MODEL)


def build_extraction_chain():
    """LCEL: prompt | model.with_structured_output(schema). Replaces the
    old hand-written "return ONLY valid JSON" prompt + json.loads() parsing
    with LangChain's structured-output mechanism — the model is constrained
    (via function-calling/JSON-schema, depending on provider support) to
    return something matching LocationExtraction, so there's no brittle
    string-matching or manual parsing/error-handling for malformed JSON."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", _EXTRACTION_SYSTEM_PROMPT + "\n{recent_turns}"),
        ("human", "{query}"),
    ])
    return prompt | build_chat_model().with_structured_output(LocationExtraction)


# AI agent that processes user queries
class WeatherAgent(BaseAgent):

    # Weather API function
    def get_weather(self, location):
        url = "http://api.weatherapi.com/v1/current.json"

        params = {
            "key": WEATHER_API_KEY,
            "q": location,
            "aqi": "no"
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if response.status_code != 200:
                if "error" in data:
                    raise Exception(data["error"]["message"])
                else:
                    raise Exception("Weather API request failed.")

            return data

        except requests.exceptions.Timeout:
            raise Exception("Network timeout. Please try again.")

        except requests.exceptions.ConnectionError:
            raise Exception("Unable to connect to Weather API.")

        except requests.exceptions.HTTPError:
            raise Exception("Weather API request failed.")

        except Exception as e:
            raise Exception(f"Weather API error: {str(e)}")

    def format_weather(self, weather):

        current = weather["current"]
        location = weather["location"]

        condition = current["condition"]["text"]
        emoji = emoji_map.get(condition, "🌤️")

        return f"""
    {emoji} Weather Report

    📍 Location:
    {location["name"]}, {location["country"]}

    🌡 Temperature:
    • {current["temp_c"]}°C
    • {current["temp_f"]}°F

    🤗 Feels Like:
    • {current["feelslike_c"]}°C
    • {current["feelslike_f"]}°F

    🌥 Condition:
    {condition}

    💨 Wind:
    {current["wind_kph"]} km/h {current["wind_dir"]}

    💧 Humidity:
    {current["humidity"]}%

    🕒 Last Updated:
    {current["last_updated"]}
    """

    def handle(self, user_query, turns=None, usage=None):

        if not user_query.strip():
            return "Please enter a valid query with a location."
        try:
            extraction = build_extraction_chain().invoke(
                {"query": user_query, "recent_turns": format_recent_turns(turns)},
                config=_usage_config(usage),
            )

            location = extraction.location
            if location in weather_cache:
                logger.info("Weather cache hit")
                return weather_cache[location]
            logger.info("Location from LLM: {}".format(location))
            if not location:
                return (
                    "❌ I couldn't determine the location.\n\n"
                    "Examples:\n"
                    "• Weather in Singapore\n"
                    "• Weather in Tokyo"
                )
            weather = self.get_weather(location)
            format_weather = self.format_weather(weather)
            weather_cache[location] = format_weather
            return format_weather

        except Exception as e:
            logger.exception("WeatherAgent failed")
            message = str(e).lower()
            if "no matching location found" in message:
                return (
                    f"❌ I couldn't find weather information for \"{location}\".\n\n"
                    "Please check the location name and try again.\n\n"
                    "Examples:\n"
                    "• Singapore\n"
                    "• Kuala Lumpur\n"
                    "• Tokyo"
                )

            return (
                "❌ Unable to retrieve weather information at the moment.\n\n"
                "Please try again later."
            )
