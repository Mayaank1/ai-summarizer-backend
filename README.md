---
title: AI Summarizer Backend
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AI Summarizer Backend

Flask API that summarizes YouTube videos / uploaded MP4s with Google Gemini, answers follow-up questions about the summary, and generates highlight clips using embedding-based key-moment detection.

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
| `BASE_URL` | yes (deployed) | Public URL of this server, used to build clip links, e.g. `https://<hf-user>-ai-summarizer-backend.hf.space` |
| `GEMINI_MODEL` | no | Defaults to `gemini-2.5-pro` |

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

## Deploy to Hugging Face Spaces

1. Create a Space at https://huggingface.co/new-space: name `ai-summarizer-backend`, SDK **Docker**, blank template, free CPU hardware.
2. In the Space's **Settings → Variables and secrets**, add secret `GENAI_API_KEY` and variable `BASE_URL` (`https://<hf-user>-ai-summarizer-backend.hf.space`).
3. Push this repo to the Space, either directly:
   ```bash
   git remote add space https://huggingface.co/spaces/<hf-user>/ai-summarizer-backend
   git push space main
   ```
   or automatically on every push to GitHub via the included workflow: add a GitHub repo secret `HF_TOKEN` (a Hugging Face write token) and a repo variable `HF_SPACE` (`<hf-user>/ai-summarizer-backend`).

The free Space sleeps after 48 hours without traffic and wakes on the next request. Its disk is ephemeral, so generated clips and chat sessions are lost on restart.
