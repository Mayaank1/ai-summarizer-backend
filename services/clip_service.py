"""
Clip generation service.
Single Responsibility: Orchestrate video clip creation from YouTube URL.
Delegates to VideoSummarizer for core logic.
"""
from pathlib import Path
from typing import Optional, Tuple

from app import VideoSummarizer
from config import Config
from logger import get_logger

logger = get_logger()


class ClipService:
    """Generates highlight clips from YouTube videos."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        transcript_service=None,
        summary_service=None,
    ):
        self._output_dir = output_dir or Config.OUTPUT_FOLDER
        self._transcript_service = transcript_service
        self._summary_service = summary_service

    def generate_clip(self, url: str, duration: int = 60) -> Tuple[Optional[Path], str]:
        """
        Generate highlight clip from YouTube URL.
        Returns (output_path, transcript_of_clip) for consistent summarization.
        """
        summarizer = VideoSummarizer(
            output_dir=self._output_dir,
            transcript_service=self._transcript_service,
            summary_service=self._summary_service,
        )
        result_path, clip_transcript = summarizer.generate_highlight_video(url, duration)
        return result_path, clip_transcript or ""
