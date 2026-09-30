"""
Centralized YouTube download service.
Single Responsibility: Download video and/or subtitles with full fallback chain.
Used by both summarization (TranscriptService) and clip generation (VideoSummarizer).
"""
import glob
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import yt_dlp

from config import Config
from logger import get_logger

logger = get_logger()

_cookie_copy: Optional[str] = None

# Cap resolution: smaller downloads and cheaper ffmpeg re-encoding on low-memory hosts
_VIDEO_FORMAT = "best[ext=mp4][height<=480]/best[ext=mp4]/best"


def _get_cookie_file() -> Optional[str]:
    """
    Writable copy of the configured cookies file, or None if not configured.
    yt-dlp writes cookies back on exit, and Render secret files are read-only.
    """
    global _cookie_copy
    if _cookie_copy and os.path.isfile(_cookie_copy):
        return _cookie_copy
    source = Config.YTDLP_COOKIES_FILE
    if not source or not os.path.isfile(source):
        return None
    _cookie_copy = os.path.join(tempfile.gettempdir(), "yt_cookies.txt")
    shutil.copyfile(source, _cookie_copy)
    logger.info("Using YouTube cookies from %s", source)
    return _cookie_copy


def with_cookies(ydl_opts: dict) -> dict:
    """Add the YouTube cookie file to yt-dlp options when one is configured."""
    cookie_file = _get_cookie_file()
    return {**ydl_opts, "cookiefile": cookie_file} if cookie_file else ydl_opts


def download_youtube(
    url: str,
    out_dir: str,
    transcript_service=None,
) -> Tuple[Optional[Path], Optional[Path], Optional[str], Optional[float]]:
    """
    Download YouTube video and subtitles with full fallback chain.
    Returns (video_path, subtitle_path, title, video_duration_seconds).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = out_path / f"video_{timestamp}"

    # Fallback 1: Download video + subtitles together
    video_path, subtitle_path, title, video_duration = _download_video_with_subs(
        url, base_filename
    )
    if video_path and subtitle_path:
        return video_path, subtitle_path, title, video_duration

    # Fallback 2: Video-only download (when subtitle fetch fails e.g. 429)
    if not video_path:
        video_path, title, video_duration = _download_video_only(url, base_filename)
    if not video_path:
        return None, None, None, None

    # Fallback 3: Subtitle-only download (separate request)
    if not subtitle_path and transcript_service:
        sub_path = transcript_service.get_subtitle_path_from_url(url, str(out_path))
        if sub_path:
            subtitle_path = Path(sub_path)
            logger.info("Got subtitles via subtitle-only fallback")

    if not subtitle_path:
        subtitle_path = _find_subtitle_file(base_filename)
    if not subtitle_path:
        logger.warning("No subtitles found; will use uniform chunking fallback")

    return video_path, subtitle_path, title, video_duration


def _download_video_with_subs(
    url: str, base_filename: Path
) -> Tuple[Optional[Path], Optional[Path], Optional[str], Optional[float]]:
    """Try video + subtitles in one request."""
    ydl_opts = {
        "format": _VIDEO_FORMAT,
        "outtmpl": str(base_filename) + ".%(ext)s",
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "postprocessors": [{"key": "FFmpegSubtitlesConvertor", "format": "srt"}],
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(with_cookies(ydl_opts)) as ydl:
            logger.info("Downloading video and subtitles...")
            info = ydl.extract_info(url, download=True)
            video_path = base_filename.with_suffix(f".{info['ext']}")
            title = info.get("title") or ""
            video_duration = info.get("duration")
            if not video_path.exists():
                return None, None, None, None
            subtitle_path = _find_subtitle_file(base_filename)
            return video_path, subtitle_path, title, video_duration
    except Exception as e:
        logger.warning("Video+subs download failed: %s", e)
        return None, None, None, None


def _download_video_only(
    url: str, base_filename: Path
) -> Tuple[Optional[Path], Optional[str], Optional[float]]:
    """Download video only (no subtitles) - works when subtitle fetch fails (e.g. 429)."""
    ydl_opts = {
        "format": _VIDEO_FORMAT,
        "outtmpl": str(base_filename) + ".%(ext)s",
        "writesubtitles": False,
        "writeautomaticsub": False,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(with_cookies(ydl_opts)) as ydl:
            logger.info("Downloading video only (subtitle fallback)...")
            info = ydl.extract_info(url, download=True)
            video_path = base_filename.with_suffix(f".{info['ext']}")
            title = info.get("title") or ""
            video_duration = info.get("duration")
            if video_path.exists():
                return video_path, title, video_duration
    except Exception as e:
        logger.error("Video-only download failed: %s", e)
    return None, None, None


def _find_subtitle_file(base_filename: Path) -> Optional[Path]:
    """Find subtitle file saved by yt-dlp (may be .en.srt, .en-US.srt, etc.)."""
    stem = str(base_filename)
    for path in glob.glob(f"{stem}*"):
        if path.lower().endswith((".srt", ".vtt")):
            return Path(path)
    return None


def fetch_youtube_metadata(url: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Lightweight metadata only (no download). For UI validation of clip length vs video duration.
    Returns (duration_seconds, title).
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(with_cookies(ydl_opts)) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("duration"), info.get("title")
    except Exception as e:
        logger.warning("YouTube metadata fetch failed: %s", e)
        return None, None
