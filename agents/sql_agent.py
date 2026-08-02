import sqlite3
import time

from tabulate import tabulate

from agents.base_agent import BaseAgent
from config import DATABASE_PATH
from core.llm_service import LLMService
from logger import logger


def get_schema():
    conn = sqlite3.connect(DATABASE_PATH)

    cursor = conn.cursor()

    schema = ""

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )

    tables = cursor.fetchall()

    for table in tables:

        table_name = table[0]

        schema += f"Table: {table_name}\n"

        cursor.execute(f"PRAGMA table_info({table_name})")

        for column in cursor.fetchall():
            schema += f"- {column[1]} ({column[2]})\n"

        schema += "\n"

    conn.close()

    return schema


def sanitize_question(question):
    return question.replace(";", "").strip()


def generate_sql(question):
    sql = LLMService.chat(
        system_prompt=f"""
    You are a SQL expert.

    Schema:

    {get_schema()}

    Return ONLY SQL.

    Only SELECT statements are allowed.
    """,
        user_prompt=question
    )

    # Remove any markdown code block syntax
    sql = sql.replace('```sql', '').replace('```SQL', '').replace('```', '')

    # Remove any explanatory text before or after the SQL 
    sql_lines = [line.strip() for line in sql.split('\n') if line.strip()]
    sql = ' '.join(sql_lines)
    sql_stripped = sql.strip().lower()
    if not (sql_stripped.startswith("select") or sql_stripped.startswith("with")):
        raise ValueError("Unable to generate a valid SQL SELECT query.")

    return sql


def validate_sql(sql):
    # Basic safety checks 
    sql_lower = sql.lower()
    if any(word in sql_lower for word in ['drop', 'delete', 'update', 'insert', 'alter', 'truncate']):
        raise ValueError("Only SELECT queries are allowed")

    if len(sql) > 500:
        raise ValueError("Query too complex")

    return sql


def execute_query(sql):
    sql = validate_sql(sql)
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        start = time.time()
        cursor = conn.cursor()
        cursor.execute(sql)
        headers = [description[0] for description in cursor.description]
        results = cursor.fetchmany(100)
        end = time.time()
        logger.info(f"Execution Time: {(end - start):.5f} seconds")
        return headers, results
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        conn.close()


def explain_query(sql):
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("EXPLAIN QUERY PLAN " + sql)
        return cursor.fetchall()


def performance_tip(sql):
    sql = sql.lower()

    if "join" in sql:
        return "Suggestion: Add indexes on JOIN columns."

    elif "order by" in sql:
        return "Suggestion: Consider indexing ORDER BY columns."

    elif "where" not in sql and "avg" not in sql and "count" not in sql:
        return "Suggestion: Use WHERE clause to reduce scanned rows."

    return "Query looks efficient."


def format_results(data):
    headers, results = data

    if not results:
        return (
            "❌ No matching records were found.\n\n"
            "Your query didn't match any data in the database.\n\n"
            "Try:\n"
            "• Check the spelling of the name\n"
            "• Use a different keyword\n"
            "• Request all records (e.g., \"List all employees\")"
        )

    return tabulate(results, headers=headers, tablefmt="grid")


class SQLAgent(BaseAgent):

    def handle(self, question):

        try:
            # Generate SQL
            question = sanitize_question(question)
            sql = generate_sql(question)
            logger.info(f"Generated SQL: {sql}")

            # Execute and format results
            data = execute_query(sql)
            # If execute_query returned an error message
            if isinstance(data, str):
                return data
            headers, results = data

            logger.info(performance_tip(sql))

            logger.info("Query Plan:")
            for row in explain_query(sql):
                logger.info(row)

            return format_results((headers, results))
        except Exception as e:
            logger.exception("SQLAgent failed")

            message = str(e).lower()

            if "no such table" in message:
                return (
                    "❌ The requested table does not exist.\n\n"
                    "Please try a different query."
                )

            if "syntax" in message:
                return (
                    "❌ I couldn't understand your database request.\n\n"
                    "Please rephrase your question."
                )

            return (
                "❌ Unable to execute the database query.\n\n"
                "Please try again later."
            )
