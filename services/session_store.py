"""
Session/store for summary persistence.
Single Responsibility: Only handles save/load of summary.
Supports session_id for multi-user isolation.
"""
import os
import re
from typing import Optional

from logger import get_logger
from .interfaces import SessionStore

logger = get_logger()


def _sanitize_session_id(session_id: Optional[str]) -> str:
    """Sanitize session_id for safe file naming."""
    if not session_id:
        return "default"
    return re.sub(r"[^\w\-]", "", session_id)[:64] or "default"


class FileSessionStore(SessionStore):
    """File-based implementation - one file per session for multi-user support."""

    def __init__(self, base_path: str = "summary"):
        self._base_path = base_path.rstrip(".txt")

    def _get_file_path(self, session_id: Optional[str]) -> str:
        sid = _sanitize_session_id(session_id)
        return "summary.txt" if sid == "default" else f"{self._base_path}_{sid}.txt"

    def save_summary(self, summary: str, session_id: Optional[str] = None) -> None:
        """Persist summary for session."""
        path = self._get_file_path(session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(summary)
        except OSError as e:
            logger.error("Failed to save summary: %s", e)
            raise

    def get_summary(self, session_id: Optional[str] = None) -> Optional[str]:
        """Retrieve summary for session."""
        path = self._get_file_path(session_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            logger.error("Failed to read summary: %s", e)
            return None
