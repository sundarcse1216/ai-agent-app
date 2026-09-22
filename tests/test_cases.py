"""
Unit tests for Smart AI Multi-Agent Assistant
Run with: python -m pytest tests/ -v
      or: python -m unittest tests.test_cases -v
"""
import json
import os
import sqlite3
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------

class TestIntentClassifier(unittest.TestCase):

    def setUp(self):
        from core.intent_classifier import IntentClassifier
        self.classifier = IntentClassifier()

    # --- happy paths ---

    def test_weather_keywords(self):
        cases = [
            "What is the weather in Singapore?",
            "Will it rain tomorrow?",
            "What is the temperature today?",
            "Is it cloudy in London?",
            "Check the humidity in Bangkok",
        ]
        for q in cases:
            with self.subTest(q=q):
                self.assertEqual(self.classifier.classify(q), "weather")

    def test_sql_keywords(self):
        cases = [
            "Show all employees",
            "What is the salary of John?",
            "Display all departments",
            "Select records from the database",
        ]
        for q in cases:
            with self.subTest(q=q):
                self.assertEqual(self.classifier.classify(q), "sql")

    def test_rag_keywords(self):
        cases = [
            "Summarize the contract",
            "What are the kitchen fittings?",
            "Explain the specification document",
            "What does the builder manual say?",
        ]
        for q in cases:
            with self.subTest(q=q):
                self.assertEqual(self.classifier.classify(q), "rag")

    def test_recommendation_keywords(self):
        cases = [
            "Recommend an event in Singapore",
            "Suggest a restaurant nearby",
            "Any movie suggestions for tonight?",
        ]
        for q in cases:
            with self.subTest(q=q):
                self.assertEqual(self.classifier.classify(q), "recommendation")

    def test_image_keywords(self):
        cases = [
            "Generate an image of a cat",
            "Draw a futuristic city",
            "Create a logo for my company",
            "Make a poster for the event",
        ]
        for q in cases:
            with self.subTest(q=q):
                self.assertEqual(self.classifier.classify(q), "image")

    def test_chat_fallback(self):
        cases = [
            "Hello",
            "What is 2 + 2?",
            "Tell me a joke",
            "How are you?",
        ]
        for q in cases:
            with self.subTest(q=q):
                self.assertEqual(self.classifier.classify(q), "chat")

    # --- edge cases ---

    def test_case_insensitive(self):
        self.assertEqual(self.classifier.classify("WEATHER IN SINGAPORE"), "weather")
        self.assertEqual(self.classifier.classify("SHOW ALL EMPLOYEES"), "sql")

    def test_empty_query_falls_back_to_chat(self):
        self.assertEqual(self.classifier.classify(""), "chat")

    def test_list_does_not_route_to_sql(self):
        # "list" was removed from sql keywords to avoid misrouting
        result = self.classifier.classify("list the best events")
        self.assertNotEqual(result, "sql")

    def test_explain_does_not_route_to_rag_for_weather(self):
        # "explain" was removed from rag keywords
        result = self.classifier.classify("explain the weather forecast")
        self.assertEqual(result, "weather")


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

class TestValidateSQL(unittest.TestCase):

    def test_valid_select(self):
        from agents.sql_agent import validate_sql
        sql = "SELECT * FROM employees"
        self.assertEqual(validate_sql(sql), sql)

    def test_valid_with_cte(self):
        from agents.sql_agent import validate_sql
        sql = "WITH cte AS (SELECT id FROM employees) SELECT * FROM cte"
        self.assertEqual(validate_sql(sql), sql)

    def test_blocks_drop(self):
        from agents.sql_agent import validate_sql
        with self.assertRaises(ValueError):
            validate_sql("DROP TABLE employees")

    def test_blocks_delete(self):
        from agents.sql_agent import validate_sql
        with self.assertRaises(ValueError):
            validate_sql("DELETE FROM employees WHERE id = 1")

    def test_blocks_update(self):
        from agents.sql_agent import validate_sql
        with self.assertRaises(ValueError):
            validate_sql("UPDATE employees SET salary = 0")

    def test_blocks_insert(self):
        from agents.sql_agent import validate_sql
        with self.assertRaises(ValueError):
            validate_sql("INSERT INTO employees VALUES (1, 'x', 'HR', 50000)")

    def test_blocks_alter(self):
        from agents.sql_agent import validate_sql
        with self.assertRaises(ValueError):
            validate_sql("ALTER TABLE employees ADD COLUMN age INT")

    def test_blocks_truncate(self):
        from agents.sql_agent import validate_sql
        with self.assertRaises(ValueError):
            validate_sql("TRUNCATE TABLE employees")

    def test_blocks_overly_long_query(self):
        from agents.sql_agent import validate_sql
        with self.assertRaises(ValueError):
            validate_sql("SELECT " + "a, " * 200)


class TestSanitizeQuestion(unittest.TestCase):

    def test_removes_semicolons(self):
        from agents.sql_agent import sanitize_question
        self.assertEqual(sanitize_question("List employees;"), "List employees")

    def test_strips_whitespace(self):
        from agents.sql_agent import sanitize_question
        self.assertEqual(sanitize_question("  List employees  "), "List employees")

    def test_multiple_semicolons(self):
        from agents.sql_agent import sanitize_question
        self.assertEqual(sanitize_question("drop; delete;"), "drop delete")


class TestPerformanceTip(unittest.TestCase):

    def test_join_tip(self):
        from agents.sql_agent import performance_tip
        tip = performance_tip("SELECT * FROM employees JOIN departments ON id = dept_id")
        self.assertIn("JOIN", tip)

    def test_order_by_tip(self):
        from agents.sql_agent import performance_tip
        tip = performance_tip("SELECT * FROM employees ORDER BY salary")
        self.assertIn("ORDER BY", tip)

    def test_no_where_tip(self):
        from agents.sql_agent import performance_tip
        tip = performance_tip("SELECT name FROM employees")
        self.assertIn("WHERE", tip)

    def test_count_is_efficient(self):
        from agents.sql_agent import performance_tip
        tip = performance_tip("SELECT COUNT(*) FROM employees")
        self.assertEqual(tip, "Query looks efficient.")

    def test_avg_is_efficient(self):
        from agents.sql_agent import performance_tip
        tip = performance_tip("SELECT AVG(salary) FROM employees")
        self.assertEqual(tip, "Query looks efficient.")


class TestFormatResults(unittest.TestCase):

    def test_empty_results_message(self):
        from agents.sql_agent import format_results
        result = format_results((["id", "name"], []))
        self.assertIn("No matching records", result)

    def test_data_appears_in_output(self):
        from agents.sql_agent import format_results
        result = format_results((["id", "name"], [(1, "John"), (2, "Jane")]))
        self.assertIn("John", result)
        self.assertIn("Jane", result)


# ---------------------------------------------------------------------------
# Image agent style detection
# ---------------------------------------------------------------------------

class TestDetectStyle(unittest.TestCase):

    def test_anime(self):
        from agents.image_agent import detect_style
        self.assertEqual(detect_style("Draw an anime character"), "anime")
        self.assertEqual(detect_style("Studio Ghibli forest scene"), "anime")
        self.assertEqual(detect_style("manga style illustration"), "anime")

    def test_cyberpunk(self):
        from agents.image_agent import detect_style
        self.assertEqual(detect_style("Cyberpunk city at night"), "cyberpunk")
        self.assertEqual(detect_style("Futuristic neon skyline"), "cyberpunk")
        self.assertEqual(detect_style("sci-fi space station"), "cyberpunk")

    def test_watercolor(self):
        from agents.image_agent import detect_style
        self.assertEqual(detect_style("A watercolor landscape"), "watercolor")
        self.assertEqual(detect_style("Soft brush painting"), "watercolor")

    def test_default_realistic(self):
        from agents.image_agent import detect_style
        self.assertEqual(detect_style("A photo of a mountain"), "realistic")
        self.assertEqual(detect_style("Generate an image of a dog"), "realistic")


# ---------------------------------------------------------------------------
# ResponseFormatter
# ---------------------------------------------------------------------------

class TestResponseFormatter(unittest.TestCase):

    def test_success_contains_agent_name(self):
        from core.response_formatter import ResponseFormatter
        result = ResponseFormatter.success("weather", "It is sunny.", 1.23)
        self.assertIn("Weather Assistant", result)
        self.assertIn("It is sunny.", result)
        self.assertIn("1.23", result)

    def test_success_all_known_agents(self):
        from core.response_formatter import ResponseFormatter
        known = ["weather", "sql", "recommendation", "rag", "image", "chat"]
        for agent in known:
            with self.subTest(agent=agent):
                result = ResponseFormatter.success(agent, "ok", 0.1)
                self.assertIn("ok", result)

    def test_success_unknown_agent_still_works(self):
        from core.response_formatter import ResponseFormatter
        result = ResponseFormatter.success("unknown_agent", "response text", 0.5)
        self.assertIn("response text", result)

    def test_error_contains_message(self):
        from core.response_formatter import ResponseFormatter
        result = ResponseFormatter.error("Something went wrong")
        self.assertIn("Something went wrong", result)
        self.assertIn("Error", result)


# ---------------------------------------------------------------------------
# EventRepository (in-memory SQLite)
# ---------------------------------------------------------------------------

class TestEventRepository(unittest.TestCase):

    def setUp(self):
        from database.event_repository import EventRepository
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
            CREATE TABLE events (
                id          INTEGER PRIMARY KEY,
                name        TEXT,
                type        TEXT,
                description TEXT,
                location    TEXT,
                date        TEXT,
                price       REAL
            )
        """)
        self.conn.executemany(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?)",
            [
                (1, "Summer Concert", "outdoor", "Live music",  "Central Park",  "2026-07-25", 25.00),
                (2, "Art Show",       "indoor",  "Modern art",  "City Gallery",  "2026-07-25", 50.00),
                (3, "Food Fest",      "outdoor", "Food event",  "Waterfront",    "2026-07-26", 10.00),
            ]
        )
        self.conn.commit()

        self.repo = EventRepository()
        self.repo._connect = lambda: self.conn

    def tearDown(self):
        self.conn.close()

    def test_get_all_events(self):
        results = self.repo.get_all_events()
        self.assertEqual(len(results), 3)

    def test_get_events_by_date_returns_correct_count(self):
        results = self.repo.get_events("2026-07-25")
        self.assertEqual(len(results), 2)

    def test_get_events_by_date_no_match(self):
        results = self.repo.get_events("2099-01-01")
        self.assertEqual(len(results), 0)

    def test_get_events_by_type_outdoor(self):
        results = self.repo.get_events_by_type("2026-07-25", "outdoor")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Summer Concert")

    def test_get_events_by_type_indoor(self):
        results = self.repo.get_events_by_type("2026-07-25", "indoor")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Art Show")

    def test_row_fields_accessible_by_name(self):
        results = self.repo.get_events("2026-07-25")
        row = results[0]
        self.assertIn(row["type"], ["outdoor", "indoor"])
        self.assertIsNotNone(row["price"])


# ---------------------------------------------------------------------------
# WeatherAgent (mocked)
# ---------------------------------------------------------------------------

MOCK_WEATHER_RESPONSE = {
    "location": {"name": "Singapore", "country": "Singapore"},
    "current": {
        "temp_c": 30, "temp_f": 86,
        "feelslike_c": 33, "feelslike_f": 91.4,
        "condition": {"text": "Sunny"},
        "wind_kph": 15, "wind_dir": "NE",
        "humidity": 80, "last_updated": "2026-07-25 10:00",
    }
}


class TestWeatherAgent(unittest.TestCase):

    @patch("agents.weather_agent.requests.get")
    def test_get_weather_success(self, mock_get):
        from agents.weather_agent import WeatherAgent
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = MOCK_WEATHER_RESPONSE
        result = WeatherAgent().get_weather("Singapore")
        self.assertEqual(result["location"]["name"], "Singapore")
        self.assertEqual(result["current"]["temp_c"], 30)

    @patch("agents.weather_agent.requests.get")
    def test_get_weather_timeout(self, mock_get):
        import requests as req
        from agents.weather_agent import WeatherAgent
        mock_get.side_effect = req.exceptions.Timeout
        with self.assertRaises(Exception) as ctx:
            WeatherAgent().get_weather("Singapore")
        self.assertIn("timeout", str(ctx.exception).lower())

    @patch("agents.weather_agent.requests.get")
    def test_get_weather_connection_error(self, mock_get):
        import requests as req
        from agents.weather_agent import WeatherAgent
        mock_get.side_effect = req.exceptions.ConnectionError
        with self.assertRaises(Exception) as ctx:
            WeatherAgent().get_weather("Singapore")
        self.assertIn("connect", str(ctx.exception).lower())

    @patch("agents.weather_agent.requests.get")
    def test_get_weather_api_error(self, mock_get):
        from agents.weather_agent import WeatherAgent
        mock_get.return_value.status_code = 400
        mock_get.return_value.json.return_value = {
            "error": {"message": "No matching location found."}
        }
        with self.assertRaises(Exception) as ctx:
            WeatherAgent().get_weather("InvalidXYZ")
        self.assertIn("No matching location found", str(ctx.exception))

    @staticmethod
    def _mock_extraction_chain(location):
        """WeatherAgent's LLM boundary is now build_extraction_chain()
        (LangChain prompt | model.with_structured_output(LocationExtraction)
        LCEL chain), not LLMService.chat — mock at that seam instead so
        these tests don't make a real LLM call."""
        from agents.weather_agent import LocationExtraction
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = LocationExtraction(location=location)
        return mock_chain

    @patch("agents.weather_agent.build_extraction_chain")
    @patch("agents.weather_agent.requests.get")
    def test_handle_no_location_extracted(self, mock_get, mock_build_chain):
        from agents.weather_agent import WeatherAgent
        mock_build_chain.return_value = self._mock_extraction_chain(None)
        result = WeatherAgent().handle("what is the weather?")
        self.assertIn("couldn't determine", result)
        mock_get.assert_not_called()

    @patch("agents.weather_agent.build_extraction_chain")
    @patch("agents.weather_agent.requests.get")
    def test_handle_empty_query(self, mock_get, mock_build_chain):
        from agents.weather_agent import WeatherAgent
        result = WeatherAgent().handle("   ")
        self.assertIn("valid query", result)
        mock_build_chain.assert_not_called()

    @patch("agents.weather_agent.build_extraction_chain")
    @patch("agents.weather_agent.requests.get")
    def test_handle_valid_city(self, mock_get, mock_build_chain):
        from agents.weather_agent import WeatherAgent
        mock_build_chain.return_value = self._mock_extraction_chain("Singapore")
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = MOCK_WEATHER_RESPONSE
        result = WeatherAgent().handle("Weather in Singapore")
        self.assertIn("Singapore", result)
        self.assertIn("30", result)

    @patch("agents.weather_agent.build_extraction_chain")
    @patch("agents.weather_agent.requests.get")
    def test_handle_uses_cache_on_repeat(self, mock_get, mock_build_chain):
        from agents.weather_agent import WeatherAgent, weather_cache
        weather_cache.clear()
        mock_build_chain.return_value = self._mock_extraction_chain("Singapore")
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = MOCK_WEATHER_RESPONSE
        agent = WeatherAgent()
        agent.handle("Weather in Singapore")
        agent.handle("Weather in Singapore")
        # API should only be called once; second call served from cache
        self.assertEqual(mock_get.call_count, 1)


# ---------------------------------------------------------------------------
# SQLAgent (mocked LLM, real SQLite)
# ---------------------------------------------------------------------------

class TestSQLAgent(unittest.TestCase):

    def setUp(self):
        from database.setup_company_database import setup_company_database
        setup_company_database()

    @patch("agents.sql_agent.LLMService.chat")
    def test_handle_valid_select(self, mock_llm):
        from agents.sql_agent import SQLAgent
        mock_llm.return_value = "SELECT * FROM employees"
        result = SQLAgent().handle("List all employees")
        self.assertIn("John", result)
        self.assertIn("Jane", result)

    @patch("agents.sql_agent.LLMService.chat")
    def test_handle_llm_returns_non_select(self, mock_llm):
        from agents.sql_agent import SQLAgent
        mock_llm.return_value = "I cannot help with that."
        result = SQLAgent().handle("do something bad")
        self.assertIn("❌", result)

    @patch("agents.sql_agent.LLMService.chat")
    def test_handle_nonexistent_table(self, mock_llm):
        from agents.sql_agent import SQLAgent
        mock_llm.return_value = "SELECT * FROM ghost_table"
        result = SQLAgent().handle("query a missing table")
        self.assertIn("❌", result)

    @patch("agents.sql_agent.LLMService.chat")
    def test_handle_dangerous_sql_blocked(self, mock_llm):
        from agents.sql_agent import SQLAgent
        mock_llm.return_value = "DELETE FROM employees"
        result = SQLAgent().handle("delete all employees")
        self.assertIn("❌", result)

    @patch("agents.sql_agent.LLMService.chat")
    def test_handle_with_cte(self, mock_llm):
        from agents.sql_agent import SQLAgent
        mock_llm.return_value = (
            "WITH top AS (SELECT * FROM employees ORDER BY salary DESC LIMIT 1) "
            "SELECT * FROM top"
        )
        result = SQLAgent().handle("Who has the highest salary?")
        self.assertNotIn("❌", result)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
