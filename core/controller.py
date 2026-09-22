import time

from core.intent_classifier import IntentClassifier
from core.response_formatter import ResponseFormatter
from core.router import Router, Session
from logger import logger
from ui import Spinner


class Controller:

    def __init__(self):
        self.classifier = IntentClassifier()
        self.router = Router()
        self.session = Session()

    def process(self, query):
        logger.info(f"User Query: {query}")

        start = time.time()
        intent = self.classifier.classify(query)
        logger.info(f"Detected Intent: {intent}")

        spinner = Spinner(ResponseFormatter.LOADING_MESSAGES[intent])
        spinner.start()
        response = self.router.route(intent, query, self.session)
        spinner.stop()

        logger.info("Response generated successfully.")
        elapsed = time.time() - start

        return ResponseFormatter.success(
            intent,
            response,
            elapsed
        )
