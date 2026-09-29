"""
AI summarization service.
Single Responsibility: Generate summaries via LLM only.
Includes retry with exponential backoff for rate limits.
"""
import re
from typing import Optional

import google.generativeai as genai

from config import Config
from logger import get_logger
from .interfaces import SummaryProvider
from .gemini_retry import with_gemini_retry

logger = get_logger()

_INTRO_STRIP = (
    r"^here'?s?\s+a\s+summary\s+of\s+the\s+video\s*:?\s*",
    r"^here'?s?\s+the\s+summary\s*:?\s*",
    r"^here\s+is\s+a\s+summary\s+(?:of\s+(?:the\s+)?(?:video|transcript)\s*)?:?\s*",
    r"^summary\s*:?\s*",
    r"^the\s+following\s+is\s+a\s+summary\s*:?\s*",
)


_TOPIC_DERIVE_PROMPT = """You help score which parts of a video transcript matter most.
Given the YouTube video title (may be empty, clickbait, or vague) and an excerpt from the START of the transcript, output ONE short line (at most 40 words) describing the actual main subject matter for semantic search against subtitle lines.
Use concrete themes and entities from the excerpt. Do not say "this video", "the video is about", or similar. No bullet points. Output only that single line.

Video title: {title}

Transcript excerpt (beginning of video):
{excerpt}
"""


def _normalize_topic_derive_output(raw: str) -> str:
    if not raw:
        return ""
    t = raw.strip()
    for p in (
        r"^here'?s?\s+(?:the\s+)?(?:topic|subject|line)\s*:?\s*",
        r"^(?:topic|subject)\s*:?\s*",
        r"^output\s*:?\s*",
    ):
        t = re.sub(p, "", t, flags=re.IGNORECASE)
    t = t.strip().strip('"').strip("'").strip()
    t = " ".join(t.split())
    if len(t) > 500:
        t = t[:497] + "..."
    return t


def _normalize_summary_output(text: str) -> str:
    """Remove common LLM preambles; keep body as-is (bullets normalized in UI or by model)."""
    if not text:
        return ""
    t = text.strip()
    for pattern in _INTRO_STRIP:
        t = re.sub(pattern, "", t, flags=re.IGNORECASE)
    return t.strip()


class GeminiSummaryService(SummaryProvider):
    """Gemini-based implementation of SummaryProvider."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or Config.GENAI_API_KEY
        self._model = model or Config.GEMINI_MODEL
        genai.configure(api_key=self._api_key)

    def derive_topic_for_key_moments(
        self, video_title: str, opening_excerpt: str
    ) -> Optional[str]:
        """Gemini: one line describing real subject for embedding similarity to subtitle segments."""
        try:
            return self._derive_topic_for_key_moments_impl(video_title, opening_excerpt)
        except Exception as e:
            logger.warning("derive_topic_for_key_moments failed: %s", e)
            return None

    @with_gemini_retry
    def _derive_topic_for_key_moments_impl(
        self, video_title: str, opening_excerpt: str
    ) -> Optional[str]:
        excerpt = (opening_excerpt or "").strip()
        if not excerpt:
            return None
        title = (video_title or "").strip() or "(none)"
        prompt = _TOPIC_DERIVE_PROMPT.format(title=title, excerpt=excerpt)
        model = genai.GenerativeModel(self._model)
        response = model.generate_content(prompt)
        raw = response.text if response.text else ""
        out = _normalize_topic_derive_output(raw)
        return out if out else None

    @with_gemini_retry
    def summarize(self, text: str, language: str = "english") -> str:
        """Generate summary of text in specified language."""
        prompt = (
            f"{Config.SUMMARY_PROMPT}"
            f"Write all bullet points in {language}.\n\n"
            f"Transcript:\n"
        )
        model = genai.GenerativeModel(self._model)
        response = model.generate_content(prompt + text)
        raw = response.text if response.text else ""
        return _normalize_summary_output(raw)
