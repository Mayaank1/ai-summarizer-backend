"""
Chat Q&A service.
Single Responsibility: Answer questions based on context only.
Includes retry for rate limits.
"""
from typing import Optional

import google.generativeai as genai

from config import Config
from logger import get_logger
from .interfaces import SessionStore
from .gemini_retry import with_gemini_retry

logger = get_logger()


class ChatService:
    """Handles chat Q&A with context from stored summary."""

    def __init__(self, session_store: SessionStore):
        self._store = session_store
        genai.configure(api_key=Config.GENAI_API_KEY)

    @with_gemini_retry
    def answer(self, user_message: str, session_id: Optional[str] = None) -> Optional[str]:
        """Generate answer based on stored summary context."""
        summary = self._store.get_summary(session_id)
        if not summary:
            return None

        prompt = f"{Config.CHAT_PROMPT}\n\nContext:\n{summary}\n\nUser Question:\n{user_message}"
        model = genai.GenerativeModel(Config.GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else ""
