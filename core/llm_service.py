import ollama

from config import OLLAMA_MODEL
from exception.llm_unavailable_error import LLMUnavailableError
from logger import logger


class LLMService:

    @staticmethod
    def chat(system_prompt: str, user_prompt: str, json_mode=False):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            if json_mode:
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=messages,
                    format="json"
                )
            else:
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=messages
                )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.exception(f"LLM unavailable {e}")
            if str(e) == "LLM_UNAVAILABLE":
                raise LLMUnavailableError (
                    "❌ The AI service is currently unavailable.\n\n"
                    "Please try again in a few moments."
                ) from e
            raise
