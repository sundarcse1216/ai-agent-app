import json

import requests

from agents.base_agent import BaseAgent
from config import WEATHER_API_KEY
from core.cache import make_cache
from core.conversation_context import format_recent_turns
from core.llm_service import LLMService
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

    def handle(self, user_query, turns=None):

        if not user_query.strip():
            return "Please enter a valid query with a location."
        try:
            response = LLMService.chat(
                system_prompt=f"""
            You are an information extraction assistant.

            Extract the location the user is asking about — a city,
            region, or country (weatherapi.com accepts any of these, not
            just cities). If the latest message doesn't name one but an
            earlier part of this conversation did (including something as
            indirect as "I'm from X"), use that instead.
            {format_recent_turns(turns)}
            Return ONLY valid JSON.

            Examples:
            {{"location":"Singapore"}}
            {{"location":"Malaysia"}}

            Rules:
            - Return only JSON.
            - Never explain.
            - Never apologize.
            - If no location is found anywhere, return:

            {{"location":null}}
            """,
                user_prompt=user_query,
                json_mode=True
            )

            details = json.loads(response)
            location = details.get("location")
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
