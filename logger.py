"""
Application logger singleton.
Ensures a single configured logger instance across the codebase.
"""
import logging
import sys
from typing import Optional


class LoggerSingleton:
    """Singleton logger - one instance, shared configuration."""
    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls, name: str = "video_summarizer") -> logging.Logger:
        """Get or create the singleton logger."""
        if cls._instance is None:
            cls._instance = logging.getLogger(name)
            cls._instance.setLevel(logging.INFO)
            cls._instance.handlers.clear()

            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
            )

            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            cls._instance.addHandler(stream_handler)

            file_handler = logging.FileHandler("video_summarizer.log")
            file_handler.setFormatter(formatter)
            cls._instance.addHandler(file_handler)

        return cls._instance


def get_logger(name: str = "video_summarizer") -> logging.Logger:
    """Convenience function to get the singleton logger."""
    return LoggerSingleton.get_logger(name)
