"""
Retry decorator for Gemini API calls.
Handles 429 (rate limit) with exponential backoff.
"""
import time
from functools import wraps
from typing import Callable, TypeVar

from google.api_core.exceptions import ResourceExhausted

from config import Config
from logger import get_logger

logger = get_logger()
T = TypeVar("T")


def with_gemini_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Retry on 429 with exponential backoff."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        last_exc = None
        backoff = Config.GEMINI_INITIAL_BACKOFF
        for attempt in range(Config.GEMINI_MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except ResourceExhausted as e:
                last_exc = e
                if attempt < Config.GEMINI_MAX_RETRIES:
                    logger.warning("Gemini rate limit (attempt %d/%d), retrying in %.1fs", attempt + 1, Config.GEMINI_MAX_RETRIES, backoff)
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error("Gemini rate limit exceeded after %d retries", Config.GEMINI_MAX_RETRIES)
        raise last_exc

    return wrapper
