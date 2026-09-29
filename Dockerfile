FROM python:3.11-slim

# ffmpeg for moviepy/yt-dlp, libgl/glib for OpenCV (scenedetect)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860
WORKDIR /home/user/app

# CPU-only torch keeps the image far smaller than the default CUDA build
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding model into the image so the first clip request doesn't download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user . .

EXPOSE 7860
# Single worker: sessions and generated files live on local disk. Timeout 0: clip generation is long-running.
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 0 main:app
