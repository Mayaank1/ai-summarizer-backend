"""
Abstract interfaces for Dependency Inversion.
Depend on abstractions, not concretions - enables testing and swapping implementations.
"""
from abc import ABC, abstractmethod
from typing import Optional


class TranscriptProvider(ABC):
    """Interface for transcript extraction - Liskov: any impl can be substituted."""

    @abstractmethod
    def get_transcript_from_url(self, url: str) -> Optional[str]:
        """Extract transcript from YouTube URL."""
        pass

    @abstractmethod
    def get_transcript_from_file(self, file_path: str) -> Optional[str]:
        """Extract transcript from video file."""
        pass


class SummaryProvider(ABC):
    """Interface for AI summarization - Open/Closed: add new providers without changing consumers."""

    def derive_topic_for_key_moments(
        self, video_title: str, opening_excerpt: str
    ) -> Optional[str]:
        """Optional: refine topic string for embedding-based clip scoring. Default: no refinement."""
        return None

    @abstractmethod
    def summarize(self, text: str, language: str) -> str:
        """Generate summary of text in specified language."""
        pass


class SessionStore(ABC):
    """Interface for summary storage - Interface Segregation: small, focused contract."""

    @abstractmethod
    def save_summary(self, summary: str, session_id: Optional[str] = None) -> None:
        """Persist summary for later retrieval. session_id enables multi-user isolation."""
        pass

    @abstractmethod
    def get_summary(self, session_id: Optional[str] = None) -> Optional[str]:
        """Retrieve stored summary for session."""
        pass
