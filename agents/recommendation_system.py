import json
from datetime import datetime

from agents.base_agent import BaseAgent
from core.conversation_context import format_recent_turns
from core.llm_service import LLMService
from database.event_repository import EventRepository
from logger import logger


class RecommendationAgent(BaseAgent):

    def __init__(self, weather_agent):
        # Injected rather than constructed here, so Router wires in the
        # same pooled WeatherAgent instance it already builds for the
        # "weather" intent instead of this agent building its own second
        # one. This is the reusable pattern for any future agent-to-agent
        # dependency: agents declare what they need, Router supplies it.
        self.weather_agent = weather_agent
        self.event_repository = EventRepository()

    def handle(self, query, turns=None):
        try:
            details = self._extract_details(query, turns)

            location = details.get("location")
            date = details.get("date")

            if not location or not date:
                return (
                    "❌ I couldn't determine the location or date from your request.\n\n"
                    "Example:\n"
                    '• "Recommend an event in Singapore tomorrow."\n'
                    '• "Suggest an event in Tokyo on Friday."'
                )

            weather = self.weather_agent.get_weather(location)
            events = self.event_repository.get_events(date)

            if not events:
                return (
                    f"❌ No events are available in {location} on {date}.\n\n"
                    "Please try:\n"
                    "• A different date\n"
                    "• Another location"
                )

            context = self._build_context(location, weather, events)
            return self._generate_recommendation(context)

        except Exception as e:
            logger.exception("RecommendationAgent failed")

            return (
                "❌ Unable to generate an event recommendation at the moment.\n\n"
                "Please try again later."
            )

    def _extract_details(self, query, turns=None):

        today = datetime.today().strftime("%Y-%m-%d")

        response = LLMService.chat(
            system_prompt=f"""
            Today's date is {today}.

            Extract ONLY:
            - location
            - date

            If the latest message is missing the location or date but an
            earlier part of this conversation mentioned it, use that
            instead.
            {format_recent_turns(turns)}
            Return ONLY valid JSON.

            Return exactly in this format:

            {{"location":"Singapore","date":"YYYY-MM-DD"}}

            Rules:
            - Return only JSON.
            - Do not include markdown.
            - Do not include explanations.
            - Do not include an "error" field.
            - Do not include an "event" field.
            - If the location is missing, use null.
            - If the date is missing, use null.
            """,
            user_prompt=query,
            json_mode=True
        )

        logger.info(f"LLM Response: {response}")
        if not response:
            raise Exception("LLM returned an empty response.")

        try:
            response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON returned by LLM: {response}")
            raise Exception("Failed to extract location and date.")

    def _build_context(self, location, weather, events):

        current = weather["current"]

        # NOTE: events aren't tagged with a city/country in this app's demo
        # dataset (event["location"] below is a venue name like "Central
        # Park", not a city) — there's no way to actually filter events by
        # the requested city. Explicitly naming the city here at least
        # makes the LLM aware of and reason about what was asked, rather
        # than silently dropping it — without this, two different cities
        # with similar weather readings could produce near-identical
        # recommendations, since nothing in the prompt distinguished them.
        context = f"""
    City requested: {location}

    Weather in {location}

    Condition: {current["condition"]["text"]}

    Temperature: {current["temp_c"]}°C

    Humidity: {current["humidity"]}%

    Wind: {current["wind_kph"]} km/h

    Available Events
    (Note: these are the events on file for this date — they are not
    necessarily located in {location}. Say so if none of them are a good
    match for that city.)

    """

        for event in events:
            context += f"""
    Name: {event["name"]}
    Type: {event["type"]}
    Description: {event["description"]}
    Location: {event["location"]}
    Date: {event["date"]}
    Price: ${event["price"]}
    """

        return context

    def _generate_recommendation(self, context):

        return LLMService.chat(
            system_prompt="""
    You are an intelligent event recommendation assistant.

    Choose the best event for the requested city and its weather.

    Consider

    - The requested city and its weather/temperature
    - Indoor or Outdoor
    - Price

    Explain why, referencing the city and its weather explicitly.

    Also recommend transportation.

    Keep your answer concise.
    """,
            user_prompt=context
        )
