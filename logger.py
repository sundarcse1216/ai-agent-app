import logging
import os
from logging.handlers import TimedRotatingFileHandler

from config import LOG_PATH

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger("CapstoneLogger")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers
if not logger.handlers:
    handler = TimedRotatingFileHandler(
        filename=LOG_PATH,
        when="midnight",  # Rotate every day
        interval=1,
        backupCount=30,  # Keep last 30 days
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)
