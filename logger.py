"""
Application logger singleton.
Ensures a single configured logger instance across the codebase.
"""
import logging
import os
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


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def log_memory(stage: str) -> None:
    """Log container memory use (Linux cgroups) so out-of-memory crashes can be traced to a stage."""
    if not os.path.exists("/sys/fs/cgroup"):
        return
    used = _read_int("/sys/fs/cgroup/memory.current") or _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    peak = _read_int("/sys/fs/cgroup/memory.peak") or _read_int("/sys/fs/cgroup/memory/memory.max_usage_in_bytes")
    if used is not None:
        get_logger().info(
            "memory after %s: %d MB (peak %s MB)",
            stage, used // 2**20, peak // 2**20 if peak is not None else "?",
        )
