"""
Flask API - Thin route layer.
SOLID: Routes only handle HTTP, delegate to services.
"""
import io
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from config import Config
from logger import get_logger
from services.youtube_download_service import fetch_youtube_metadata
from services import (
    TranscriptService,
    GeminiSummaryService,
    ChatService,
    FileSessionStore,
    ClipService,
)

# --- App setup ---
app = Flask(__name__)
# Trust Render's proxy headers so request.url_root is the public https URL
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
Config.ensure_directories()

# --- Service composition (Dependency Injection) ---
session_store = FileSessionStore()
transcript_service = TranscriptService()
summary_service = GeminiSummaryService()
chat_service = ChatService(session_store=session_store)
clip_service = ClipService(
    transcript_service=transcript_service,
    summary_service=summary_service,
)
_executor = ThreadPoolExecutor(max_workers=2)

# --- Logging (singleton) ---
logger = get_logger()

MAX_UPLOAD_BYTES = Config.MAX_UPLOAD_SIZE_MB * 1024 * 1024


# --- Validation helpers ---
def _has_valid_file_extension(filename: str) -> bool:
    """Check if filename has an allowed extension (e.g. .mp4)."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def _get_last_user_message(messages: list) -> Optional[str]:
    """Extract the most recent user message from chat history."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content")
    return None


def _get_session_id() -> Optional[str]:
    """Extract session_id from JSON body or form data."""
    if request.is_json:
        return (request.get_json(silent=True) or {}).get("session_id")
    return request.form.get("session_id")


def _execute_with_timeout(func, *args, timeout: int = Config.MAX_PROCESSING_TIMEOUT, **kwargs):
    """Run long-running operation in thread pool with timeout. Returns None on timeout."""
    future = _executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        logger.warning("Operation timed out after %ds", timeout)
        return None


def _summarize_video_file(upload_path: str, language: str):
    """Transcribe and summarize video file. Returns (summary, None) or (None, error_msg)."""
    transcript = transcript_service.get_transcript_from_file(upload_path)
    if not transcript:
        return None, "Could not retrieve transcript from the video."
    summary = summary_service.summarize(transcript, language)
    return summary, None


def _summarize_video_from_url(url: str, language: str):
    """Transcribe and summarize video from YouTube URL. Returns (summary, None) or (None, error_msg)."""
    transcript = transcript_service.get_transcript_from_url(url, subtitles_only=True)
    if not transcript:
        return None, "Could not retrieve transcript. No subtitles and transcription failed."
    summary = summary_service.summarize(transcript, language)
    return summary, None


def _generate_highlight_clip(url: str, duration: int):
    """Generate highlight clip from YouTube URL. Returns (output_path, clip_transcript)."""
    return clip_service.generate_clip(url, duration)


# --- Routes (thin - delegate to services) ---
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "Backend AI is running."})


@app.route("/summarize", methods=["POST"])
def summarize():
    """Summarize video from file upload or YouTube URL."""
    if "file" in request.files:
        return _handle_summarize_file_upload()
    if request.is_json:
        return _handle_summarize_url_request()
    return jsonify({"success": False, "error": "No file or URL provided"}), 400


def _handle_summarize_file_upload():
    """Handle POST /summarize with file upload."""
    file = request.files['file']
    if not file or not _has_valid_file_extension(file.filename):
        return jsonify({"success": False, "error": "Unsupported file type. Only .mp4 allowed."}), 400

    # File size limit
    file.seek(0, io.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_UPLOAD_BYTES:
        return jsonify({
            "success": False,
            "error": f"File too large. Max {Config.MAX_UPLOAD_SIZE_MB} MB allowed."
        }), 400

    language = request.form.get("language", "english")
    session_id = _get_session_id()
    upload_path = Path(Config.UPLOAD_FOLDER) / file.filename
    file.save(str(upload_path))

    result = _execute_with_timeout(_summarize_video_file, str(upload_path), language)
    if result is None:
        return jsonify({"success": False, "error": "Processing timed out. Try a shorter video."}), 408
    summary, err = result
    if err:
        return jsonify({"success": False, "error": err}), 400

    session_store.save_summary(summary, session_id)
    return jsonify({"success": True, "summary": summary})


def _handle_summarize_url_request():
    """Handle POST /summarize with JSON body containing YouTube URL."""
    data = request.get_json() or {}
    url = data.get("url")
    language = data.get("language", "english")
    session_id = _get_session_id()

    if not url:
        return jsonify({"success": False, "error": "No URL provided"}), 400

    result = _execute_with_timeout(_summarize_video_from_url, url, language)
    if result is None:
        return jsonify({"success": False, "error": "Processing timed out."}), 408
    summary, err = result
    if err:
        return jsonify({"success": False, "error": err}), 400

    session_store.save_summary(summary, session_id)
    return jsonify({"success": True, "summary": summary})


@app.route("/youtube_metadata", methods=["POST"])
def youtube_metadata():
    """Return YouTube video duration (seconds) for clip UI validation — no download."""
    data = request.get_json() or {}
    url = data.get("url")
    if not url:
        return jsonify({"error": "YouTube video URL is required"}), 400
    duration, title = fetch_youtube_metadata(url)
    if duration is None:
        return jsonify({"error": "Could not read video length. Check the URL."}), 400
    return jsonify({
        "duration_seconds": int(round(duration)),
        "title": title or "",
    }), 200


@app.route("/summarize_video", methods=["POST"])
def handle_summarize_video():
    """Handle POST /summarize_video: generate highlight clip and return summary."""
    data = request.get_json() or {}
    url = data.get("url")
    try:
        raw_len = data.get("length", data.get("duration", 60))
        duration = int(str(raw_len).strip()) if raw_len is not None and str(raw_len).strip() != "" else 60
    except (TypeError, ValueError):
        return jsonify({"error": "Clip length must be a whole number of seconds."}), 400
    # Clip summaries always use English; language is not exposed on the clipper UI.
    language = "english"
    session_id = _get_session_id()

    if not url:
        return jsonify({"error": "YouTube video URL is required"}), 400

    if duration < Config.CLIP_MIN_SECONDS:
        return jsonify({
            "error": f"Clip length must be at least {Config.CLIP_MIN_SECONDS} seconds.",
        }), 400
    if duration > Config.CLIP_MAX_SECONDS:
        return jsonify({
            "error": f"Clip length must be at most {Config.CLIP_MAX_SECONDS} seconds.",
        }), 400

    try:
        result = _execute_with_timeout(
            _generate_highlight_clip,
            url,
            duration,
            timeout=Config.CLIP_GENERATION_TIMEOUT,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if result is None:
        return jsonify({"error": "Clip generation timed out. Try a shorter duration."}), 408

    result_path, clip_transcript = result
    if not result_path:
        return jsonify({"error": "Failed to create summary video. Check the logs."}), 500

    base_url = (Config.BASE_URL or request.url_root).rstrip("/")
    video_url = f"{base_url}/{Config.OUTPUT_FOLDER}/{result_path.name}"

    summary = ""
    if clip_transcript:
        summary = summary_service.summarize(clip_transcript, language)
        session_store.save_summary(summary, session_id)

    return jsonify({
        "message": "Summary video created successfully",
        "path": video_url,
        "summary": summary,
    }), 200


@app.route(f"/{Config.OUTPUT_FOLDER}/<path:filename>")
def serve_output_file(filename):
    """Serve generated clip files from output directory."""
    return send_from_directory(Config.OUTPUT_FOLDER, filename)


@app.errorhandler(413)
def handle_request_entity_too_large(_):
    return jsonify({
        "success": False,
        "error": f"File too large. Max {Config.MAX_UPLOAD_SIZE_MB} MB allowed."
    }), 413


@app.route("/chat", methods=["POST"])
def chat():
    if not request.is_json:
        return jsonify({"success": False, "error": "JSON body required"}), 400

    data = request.get_json()
    messages = data.get("messages", [])
    user_message = _get_last_user_message(messages)
    session_id = _get_session_id()

    if not user_message:
        return jsonify({"success": False, "error": "No user message found"}), 400

    summary = session_store.get_summary(session_id)
    if not summary:
        return jsonify({"success": False, "error": "Summary not found. Please summarize a video first."}), 400

    answer = chat_service.answer(user_message, session_id)
    return jsonify({"success": True, "answer": answer or ""})


if __name__ == "__main__":
    import os

    # Werkzeug's reloader spawns a child process that serves HTTP; the IDE debugger stays on the
    # parent, so route breakpoints never hit. Default: no reloader. Enable only when running
    # from a plain terminal: FLASK_USE_RELOADER=true python main.py
    use_reloader = os.environ.get("FLASK_USE_RELOADER", "false").lower() in ("1", "true", "yes")
    app.run(debug=True, port=5000, use_reloader=use_reloader, threaded=True)
