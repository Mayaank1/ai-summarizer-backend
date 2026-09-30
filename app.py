#!/usr/bin/env python
from __future__ import unicode_literals

import argparse
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import chardet
import pysrt

from config import Config
from logger import get_logger, log_memory
from services.key_moment_service import select_key_moments
from services.youtube_download_service import download_youtube

logger = get_logger()


def _subtitle_segment_to_time_range(segment: pysrt.SubRipItem) -> Tuple[float, float]:
    """Convert subtitle segment to (start_seconds, end_seconds)."""
    start = segment.start.hours * 3600 + segment.start.minutes * 60 + segment.start.seconds + segment.start.milliseconds / 1000
    end = segment.end.hours * 3600 + segment.end.minutes * 60 + segment.end.seconds + segment.end.milliseconds / 1000
    return start, end


def _parse_subtitle_segments(srt_file: pysrt.SubRipFile) -> List[dict]:
    """Parse SRT file into list of {text, start, end} for key moment selection."""
    segments = []
    for seg in srt_file:
        start, end = _subtitle_segment_to_time_range(seg)
        text = seg.text.replace("\n", " ").strip()
        if text:
            segments.append({"text": text, "start": start, "end": end})
    return segments


def _opening_excerpt_from_segments(
    segments: List[dict], opening_window_sec: float, max_chars: int
) -> str:
    """Concatenate subtitle text from cues overlapping [0, opening_window_sec) for Gemini topic input."""
    if not segments or opening_window_sec <= 0:
        return ""
    parts = []
    for seg in segments:
        try:
            ss = float(seg["start"])
            se = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if se <= 0 or ss >= opening_window_sec:
            continue
        if se > 0 and ss < opening_window_sec:
            t = (seg.get("text") or "").strip()
            if t:
                parts.append(t)
    text = " ".join(parts).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def _snap_regions_to_scene_cuts(
    regions: List[Tuple[float, float]], video_path: Path, tolerance: float = 2.0
) -> List[Tuple[float, float]]:
    """Optionally snap region boundaries to nearest scene cuts (when PySceneDetect available)."""
    try:
        from scenedetect import AdaptiveDetector, SceneManager, open_video

        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(AdaptiveDetector())
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        cuts = sorted({t.get_seconds() for pair in scene_list for t in pair})
        if not cuts:
            return regions
        snapped = []
        for start, end in regions:
            near_start = min(cuts, key=lambda c: abs(c - start))
            near_end = min(cuts, key=lambda c: abs(c - end))
            new_start = near_start if abs(near_start - start) <= tolerance else start
            new_end = near_end if abs(near_end - end) <= tolerance else end
            if new_end > new_start:
                snapped.append((new_start, new_end))
        return snapped if snapped else regions
    except Exception:
        return regions


def _get_transcript_for_regions(subtitle_path: Path, regions: List[Tuple[float, float]]) -> str:
    """Extract transcript text from subtitle segments overlapping the given time regions."""
    if not subtitle_path or not subtitle_path.exists():
        return ""
    try:
        with open(subtitle_path, 'rb') as f:
            enc = chardet.detect(f.read())['encoding'] or 'utf-8'
        srt_file = pysrt.open(str(subtitle_path), encoding=enc)
        text_parts = []
        for segment in srt_file:
            seg_start, seg_end = _subtitle_segment_to_time_range(segment)
            for region_start, region_end in regions:
                # Check if segment overlaps with region
                if seg_end > region_start and seg_start < region_end:
                    text_parts.append(segment.text.replace('\n', ' '))
                    break
        return ' '.join(text_parts).strip()
    except Exception as e:
        logger.error(f"Error extracting transcript for regions: {e}")
        return ""


class VideoSummarizer:
    def __init__(
        self,
        output_dir: str = "output",
        transcript_service=None,
        summary_service=None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp())
        self._transcript_service = transcript_service
        self._summary_service = summary_service

    def _check_ffmpeg_available(self) -> bool:
        """Verify FFmpeg is installed (required for video processing)."""
        dependencies = ['ffmpeg']

        for dep in dependencies:
            try:
                subprocess.run([dep, '-version'], capture_output=True)
            except FileNotFoundError:
                logger.error(f"{dep} not found. Please install it first.")
                return False
        return True

    def _select_key_regions_from_subtitles(
        self, subtitle_path: Path, duration: int = 60, topic: Optional[str] = None
    ) -> List[Tuple[float, float]]:
        """Select key time regions using embedding-based importance scoring."""
        try:
            with open(subtitle_path, "rb") as f:
                enc = chardet.detect(f.read())["encoding"] or "utf-8"

            srt_file = pysrt.open(str(subtitle_path), encoding=enc)
            if len(srt_file) == 0:
                logger.error("No subtitles found in file")
                return []

            segments = _parse_subtitle_segments(srt_file)
            logger.info("Embedding-based key moment selection for %d segments", len(segments))

            topic_for_scoring = topic.strip() if topic and topic.strip() else None
            if (
                Config.TOPIC_USE_GEMINI
                and self._summary_service is not None
                and segments
            ):
                excerpt = _opening_excerpt_from_segments(
                    segments,
                    Config.TOPIC_GEMINI_OPENING_SEC,
                    Config.TOPIC_GEMINI_MAX_EXCERPT_CHARS,
                )
                if len(excerpt) >= Config.TOPIC_GEMINI_MIN_EXCERPT_CHARS:
                    refined = self._summary_service.derive_topic_for_key_moments(
                        topic or "", excerpt
                    )
                    if refined:
                        topic_for_scoring = refined
                        preview = refined if len(refined) <= 120 else refined[:117] + "..."
                        logger.info("Gemini-derived topic for key moments: %s", preview)

            clip_window = 6.0
            regions = select_key_moments(
                segments,
                target_duration=float(duration),
                clip_window=clip_window,
                topic=topic_for_scoring,
            )

            return self._adjust_regions_to_target_duration(regions, duration)

        except Exception as e:
            logger.error("Error processing subtitles: %s", e)
            return []

    def _adjust_regions_to_target_duration(self, regions: List[Tuple[float, float]], target_duration: int) -> List[Tuple[float, float]]:
        """Extend or trim regions so total duration matches target (within 5s tolerance)."""
        if not regions:
            return []

        total_duration = sum(end - start for start, end in regions)

        if abs(total_duration - target_duration) <= 5:  # Within 5 seconds tolerance
            return regions

        if total_duration < target_duration:
            # Extend regions proportionally
            ratio = target_duration / total_duration
            optimized = []
            for start, end in regions:
                duration = end - start
                new_duration = duration * ratio
                extension = (new_duration - duration) / 2
                optimized.append((max(0, start - extension), end + extension))
            return optimized
        else:
            # Trim regions to fit target duration
            regions.sort(key=lambda x: x[1] - x[0], reverse=True)
            optimized = []
            current_duration = 0

            for start, end in regions:
                duration = end - start
                if current_duration + duration <= target_duration:
                    optimized.append((start, end))
                    current_duration += duration
                else:
                    remaining = target_duration - current_duration
                    if remaining > 0:
                        optimized.append((start, start + remaining))
                    break

            return sorted(optimized, key=lambda x: x[0])

    def _build_highlight_video(self, video_path: Path, regions: List[Tuple[float, float]], output_filename: str) -> Optional[Path]:
        """
        Cut each region with ffmpeg and join them into one highlight video.
        ffmpeg streams frames itself, so memory stays low (MoviePy decoded frames in Python and ran out of RAM).
        """
        if not regions:
            logger.error("No regions to process")
            return None

        logger.info("Creating video summary...")
        segment_paths = []
        for i, (start, end) in enumerate(regions):
            segment_path = self.temp_dir / f"segment_{i:03d}.mp4"
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{end - start:.3f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-threads", "1",
                    "-c:a", "aac", "-ar", "44100", "-ac", "2",
                    str(segment_path),
                ],
                capture_output=True, text=True,
            )
            if result.returncode != 0 or not segment_path.exists():
                logger.warning("Failed to process clip %.1f-%.1f: %s", start, end, result.stderr.strip()[-500:])
                continue
            segment_paths.append(segment_path)

        if not segment_paths:
            logger.error("No valid clips to concatenate")
            return None

        # All segments share identical encoding settings, so the concat demuxer can join them without re-encoding
        list_file = self.temp_dir / "segments.txt"
        list_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in segment_paths), encoding="utf-8")
        output_path = self.output_dir / output_filename

        logger.info(f"Writing final video to {output_path}")
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c", "copy", "-movflags", "+faststart",
                str(output_path),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not output_path.exists():
            logger.error("Error creating summary video: %s", result.stderr.strip()[-500:])
            return None
        return output_path

    def _cleanup_temp_files(self):
        """Remove temporary download directory."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {str(e)}")

    def generate_highlight_video(self, url: str, duration: int = 60) -> Tuple[Optional[Path], str]:
        """Full pipeline: download, select key moments, build highlight video. Returns (output_path, clip_transcript)."""
        try:
            if not self._check_ffmpeg_available():
                return None, ""

            video_path, subtitle_path, title, video_duration = download_youtube(
                url, str(self.temp_dir), transcript_service=self._transcript_service
            )
            log_memory("download")
            if not video_path:
                return None, ""
            if video_duration is not None and duration > video_duration:
                raise ValueError(
                    f"Requested clip duration ({duration}s) exceeds video length ({int(video_duration)}s)"
                )

            # Key moments require subtitles (no auto / uniform fallback)
            if not subtitle_path or not subtitle_path.exists():
                raise ValueError(
                    "Subtitles not found for this video. Try a video with captions enabled."
                )
            regions = self._select_key_regions_from_subtitles(
                subtitle_path, duration, topic=title
            )
            clip_transcript = _get_transcript_for_regions(subtitle_path, regions)
            log_memory("key moment selection")

            if not regions:
                return None, ""

            if Config.SNAP_TO_SCENE_CUTS:
                regions = _snap_regions_to_scene_cuts(regions, video_path)

            output_filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            result_path = self._build_highlight_video(video_path, regions, output_filename)
            log_memory("cut and join")

            self._cleanup_temp_files()
            return result_path, clip_transcript

        except ValueError:
            self._cleanup_temp_files()
            raise
        except Exception as e:
            logger.error("Error in processing pipeline: %s", e)
            self._cleanup_temp_files()
            return None, ""


def main():
    parser = argparse.ArgumentParser(
        description='Create a summary of a YouTube video')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--duration', type=int, default=60,
                        help='Target duration of the summary in seconds (default: 60)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for the summary video')
    args = parser.parse_args()

    from services import TranscriptService, GeminiSummaryService
    transcript_svc = TranscriptService()
    summary_svc = GeminiSummaryService()
    summarizer = VideoSummarizer(
        output_dir=args.output_dir,
        transcript_service=transcript_svc,
        summary_service=summary_svc,
    )
    try:
        result_path, _ = summarizer.generate_highlight_video(args.url, args.duration)
    except ValueError as e:
        print(f"\nError: {e}")
        return

    if result_path:
        print(f"\nSummary video created successfully: {result_path}")
    else:
        print("\nFailed to create summary video. Check the logs for details.")


if __name__ == "__main__":
    main()
