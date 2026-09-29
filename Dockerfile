FROM python:3.11-slim

# ffmpeg for moviepy/yt-dlp
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Deno: JS runtime yt-dlp uses to solve YouTube's player challenges
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

ENV PYTHONUNBUFFERED=1 \
    PORT=10000
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000
# Single worker: sessions and generated files live on local disk (and the free tier has 512 MB RAM).
# Timeout 0: clip generation is long-running.
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 0 main:app
