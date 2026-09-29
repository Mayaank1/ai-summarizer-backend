"""
Transcript extraction service.
Single Responsibility: Get transcript from URL or file only.
"""
import os
import re
import tempfile
from typing import Optional

import chardet
import pysrt
import yt_dlp
from moviepy import VideoFileClip

import google.generativeai as genai

from config import Config
from logger import get_logger
from .interfaces import TranscriptProvider
from .youtube_download_service import download_youtube, with_cookies

logger = get_logger()


class TranscriptService(TranscriptProvider):
    """Extracts transcripts from YouTube URLs or video files."""

    def __init__(self):
        genai.configure(api_key=Config.GENAI_API_KEY)

    def get_transcript_from_url(self, url: str, subtitles_only: bool = True) -> Optional[str]:
        """
        Get transcript from YouTube URL.
        subtitles_only=True: Download only subtitles (fast, no video). Falls back to full download if no subs.
        subtitles_only=False: Download full video (needed for clip generation).
        """
        if subtitles_only:
            subtitle_path = self._download_subtitles_only(url)
            if subtitle_path:
                return self._read_subtitle_file(subtitle_path)
            # No subtitles - fall back to full download + transcription
            subtitles_only = False

        if not subtitles_only:
            out_dir = os.path.join(os.getcwd(), Config.DOWNLOADS_FOLDER)
            video_path, subtitle_path, _, _ = download_youtube(
                url, out_dir, transcript_service=self
            )
            if not video_path or not video_path.exists():
                return None
            transcript = self._read_subtitle_file(str(subtitle_path) if subtitle_path else None)
            if transcript:
                return transcript
            return self.get_transcript_from_file(str(video_path))
        return None

    def get_transcript_from_file(self, file_path: str) -> Optional[str]:
        """Extract transcript from video file via Gemini audio transcription."""
        try:
            video = VideoFileClip(file_path)
            audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.wav")
            video.audio.write_audiofile(audio_path)

            audio_file = genai.upload_file(audio_path, mime_type="audio/wav")
            model = genai.GenerativeModel(Config.GEMINI_MODEL)
            response = model.generate_content([
                audio_file,
                "Transcribe this audio exactly. Output only the transcript text, nothing else. Preserve the original language."
            ])
            return response.text.strip() if response.text else None
        except Exception as e:
            logger.error("Error extracting transcript from video: %s", e)
            return None

    def _download_subtitles_only(self, url: str) -> Optional[str]:
        """Download only subtitles (no video) - fast path for summarization."""
        import glob
        out_dir = os.path.join(os.getcwd(), Config.DOWNLOADS_FOLDER)
        os.makedirs(out_dir, exist_ok=True)
        outtmpl = os.path.join(out_dir, "%(id)s")
        ydl_opts = {
            "skip_download": True,
            "outtmpl": outtmpl,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "srt",
            "subtitleslangs": ["en"],
            "quiet": True,
        }
        try:
            with yt_dlp.YoutubeDL(with_cookies(ydl_opts)) as ydl:
                result = ydl.extract_info(url, download=True)
                vid = result.get("id", "")
                # yt-dlp saves as {outtmpl}.{lang}.{ext} e.g. downloads/abc123.en.srt
                for ext in [".en.srt", ".en.vtt", ".srt"]:
                    path = os.path.join(out_dir, f"{vid}{ext}")
                    if os.path.isfile(path):
                        return path
                for path in glob.glob(os.path.join(out_dir, f"{vid}*")):
                    if path.endswith((".srt", ".vtt")):
                        return path
                return None
        except Exception as e:
            logger.warning("Subtitle-only download failed: %s", e)
            return None

    def get_subtitle_path_from_url(self, url: str, out_dir: str) -> Optional[str]:
        """
        Try subtitle-only download (like summarization path). Returns path if found.
        out_dir: directory to save subtitles (e.g. temp dir for clip generation).
        """
        import glob
        import os
        outtmpl = os.path.join(out_dir, "%(id)s")
        ydl_opts = {
            "skip_download": True,
            "outtmpl": outtmpl,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "srt",
            "subtitleslangs": ["en"],
            "quiet": True,
        }
        try:
            with yt_dlp.YoutubeDL(with_cookies(ydl_opts)) as ydl:
                result = ydl.extract_info(url, download=True)
                vid = result.get("id", "")
                for ext in [".en.srt", ".en.vtt", ".srt"]:
                    path = os.path.join(out_dir, f"{vid}{ext}")
                    if os.path.isfile(path):
                        return path
                for path in glob.glob(os.path.join(out_dir, f"{vid}*")):
                    if path.lower().endswith((".srt", ".vtt")):
                        return path
                return None
        except Exception as e:
            logger.warning("Subtitle-only download failed: %s", e)
            return None

    def _read_subtitle_file(self, subtitle_path: Optional[str]) -> Optional[str]:
        """Extract text from SRT/VTT file."""
        if not subtitle_path or not os.path.isfile(subtitle_path):
            return None
        try:
            with open(subtitle_path, "rb") as f:
                raw = f.read()
            enc = chardet.detect(raw)["encoding"] or "utf-8"
            text = raw.decode(enc)

            if str(subtitle_path).lower().endswith(".srt"):
                srt_file = pysrt.open(str(subtitle_path), encoding=enc)
                return " ".join(seg.text for seg in srt_file).replace("\n", " ")

            lines = text.split("\n")
            text_lines = [
                line.strip() for line in lines
                if line.strip() and not line.startswith("WEBVTT") and "-->" not in line and not re.match(r"^\d+$", line.strip())
            ]
            return " ".join(text_lines) if text_lines else None
        except Exception as e:
            logger.error("Error reading subtitle file: %s", e)
            return None
