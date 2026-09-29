# AI Summarizer Backend

Flask API that summarizes YouTube videos / uploaded MP4s with Google Gemini, answers follow-up questions about the summary, and generates highlight clips using embedding-based key-moment detection (Gemini embeddings + TextRank).

Frontend: [ai-summarizer-frontend](https://github.com/Mayaank1/ai-summarizer-frontend)

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Health check |
| POST | `/summarize` | Summarize a YouTube URL (JSON `{url, language}`) or an uploaded `.mp4` (form `file`) |
| POST | `/youtube_metadata` | Video duration and title, no download |
| POST | `/summarize_video` | Generate a highlight clip (`{url, length}`) and summarize it |
| POST | `/chat` | Ask questions about the latest summary |
| GET | `/output/<file>` | Serve generated clips |

## Environment variables

| Name | Required | Description |
|---|---|---|
| `GENAI_API_KEY` | yes | Gemini API key from https://aistudio.google.com/apikey |
| `BASE_URL` | yes (deployed) | Public URL of this server, used to build clip links, e.g. `https://ai-summarizer-backend.onrender.com` |
| `GEMINI_MODEL` | no | Defaults to `gemini-2.5-pro` |
| `EMBEDDING_MODEL` | no | Defaults to `models/gemini-embedding-001` |

See [config.py](config.py) for the rest.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
cp .env.example .env          # then fill in GENAI_API_KEY
python main.py                # http://127.0.0.1:5000
```

Requires `ffmpeg` on PATH.

## Deploy to Render (free)

1. At https://dashboard.render.com choose **New → Blueprint** and connect this repo. Render reads [render.yaml](render.yaml) and creates a free Docker web service.
2. When prompted, set `GENAI_API_KEY`, and set `BASE_URL` to the service URL (e.g. `https://ai-summarizer-backend.onrender.com`).
3. Every push to `main` redeploys automatically.

Free-tier limits: 512 MB RAM, the service sleeps after 15 minutes idle (first request afterwards takes ~1 minute), and the disk is ephemeral, so generated clips and chat sessions are lost on restart. Very long clip jobs may run out of memory.
