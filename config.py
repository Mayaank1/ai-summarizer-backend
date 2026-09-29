"""
Application configuration.
Single Responsibility: Centralize all config and env vars.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

_API_DIR = Path(__file__).resolve().parent
load_dotenv(_API_DIR / ".env")


class Config:
    """Application configuration - Open/Closed: add new config without modifying consumers."""
    GENAI_API_KEY: str = os.getenv("GENAI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "uploads")
    OUTPUT_FOLDER: str = os.getenv("OUTPUT_FOLDER", "output")
    DOWNLOADS_FOLDER: str = os.getenv("DOWNLOADS_FOLDER", "downloads")
    ALLOWED_EXTENSIONS: set = frozenset({"mp4"})
    BASE_URL: str = os.getenv("BASE_URL", "http://localhost:5000")
    # Netscape-format YouTube cookies so yt-dlp isn't bot-blocked on cloud hosts (Render secret files mount here)
    YTDLP_COOKIES_FILE: str = os.getenv("YTDLP_COOKIES_FILE", "/etc/secrets/cookies.txt")

    SUMMARY_PROMPT: str = (
        "You are a YouTube video summarizer. Summarize the transcript below in at least 200 words. "
        "Output format rules: use one bullet per line only. Each line must start with '- ' (hyphen and space). "
        "Do not include any title, preamble, or filler phrases (e.g. do not write "
        "\"Here is a summary\", \"Here's the summary\", or \"Summary:\"). Output only the bullet lines, nothing else.\n\n"
    )
    CHAT_PROMPT: str = (
        "You are a helpful AI assistant. Based on the context provided, "
        "answer the user's question accurately and concisely: "
    )

    # Embedding-based key moment detection
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    SNAP_TO_SCENE_CUTS: bool = os.getenv("SNAP_TO_SCENE_CUTS", "true").lower() in ("true", "1", "yes")
    # Gemini-derived topic for clip key-moment relevance (title + opening transcript excerpt)
    TOPIC_USE_GEMINI: bool = os.getenv("TOPIC_USE_GEMINI", "true").lower() in ("true", "1", "yes")
    TOPIC_GEMINI_OPENING_SEC: float = float(os.getenv("TOPIC_GEMINI_OPENING_SEC", "120"))
    TOPIC_GEMINI_MAX_EXCERPT_CHARS: int = int(os.getenv("TOPIC_GEMINI_MAX_EXCERPT_CHARS", "10000"))
    TOPIC_GEMINI_MIN_EXCERPT_CHARS: int = int(os.getenv("TOPIC_GEMINI_MIN_EXCERPT_CHARS", "20"))

    # Bottleneck mitigations
    MAX_PROCESSING_TIMEOUT: int = int(os.getenv("MAX_PROCESSING_TIMEOUT", "300"))  # 5 min (URL/file summarize)
    # Clip pipeline: embeddings on hundreds of subtitle segments + MoviePy can exceed 5 min.
    CLIP_GENERATION_TIMEOUT: int = int(os.getenv("CLIP_GENERATION_TIMEOUT", str(2 * 60 * 60)))  # 2 hours
    CLIP_MIN_SECONDS: int = int(os.getenv("CLIP_MIN_SECONDS", "10"))
    CLIP_MAX_SECONDS: int = int(os.getenv("CLIP_MAX_SECONDS", "600"))
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))  # 500 MB
    GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    GEMINI_INITIAL_BACKOFF: float = float(os.getenv("GEMINI_INITIAL_BACKOFF", "2.0"))

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure required directories exist."""
        for folder in (cls.UPLOAD_FOLDER, cls.OUTPUT_FOLDER, cls.DOWNLOADS_FOLDER):
            Path(folder).mkdir(parents=True, exist_ok=True)
