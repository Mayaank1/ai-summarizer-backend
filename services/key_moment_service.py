"""
Embedding-based key moment detection.
Uses Gemini embeddings + TextRank for semantic importance scoring.
"""
from typing import List, Optional, Tuple

import google.generativeai as genai
import numpy as np

from config import Config
from logger import get_logger
from .gemini_retry import with_gemini_retry

logger = get_logger()

# Gemini batchEmbedContents accepts at most 100 texts per request
_EMBED_BATCH_SIZE = 100


@with_gemini_retry
def _embed_batch(texts: List[str]) -> List[List[float]]:
    result = genai.embed_content(
        model=Config.EMBEDDING_MODEL,
        content=texts,
        task_type="semantic_similarity",
        output_dimensionality=Config.EMBEDDING_DIMENSIONS,
    )
    return result["embedding"]


def _embed(texts: List[str]) -> np.ndarray:
    """Embed texts via the Gemini API (no local model, keeps memory low)."""
    genai.configure(api_key=Config.GENAI_API_KEY)
    vectors = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        vectors.extend(_embed_batch(texts[i:i + _EMBED_BATCH_SIZE]))
    return np.asarray(vectors, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    """Pairwise cosine similarity between rows of a and rows of b (defaults to a)."""
    b = a if b is None else b
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def _compute_textrank_scores(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute TextRank-style importance scores from embedding similarity.
    Higher score = more central/salient in the document.
    """
    n = len(embeddings)
    if n <= 1:
        return np.ones(n)

    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, 0)

    # Normalize to get transition-like matrix (optional damping)
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1)
    sim = sim / row_sums

    # Power iteration for PageRank
    scores = np.ones(n) / n
    for _ in range(50):
        new_scores = 0.85 * sim.T @ scores + 0.15 / n
        if np.allclose(scores, new_scores):
            break
        scores = new_scores

    return scores


def _compute_topic_relevance_scores(embeddings: np.ndarray, topic_embedding: np.ndarray) -> np.ndarray:
    """Cosine similarity to topic - higher = more relevant to video topic."""
    return cosine_similarity(embeddings, topic_embedding).flatten()


def select_key_moments(
    segments: List[dict],
    target_duration: float,
    clip_window: float = 6.0,
    topic: Optional[str] = None,
) -> List[Tuple[float, float]]:
    """
    Select key moments using embedding-based importance scoring.

    Args:
        segments: List of {text, start, end}
        target_duration: Target total clip duration in seconds
        clip_window: Seconds around each key moment (centered)
        topic: Optional query string for relevance boosting (e.g. YouTube title or Gemini-derived topic line)

    Returns:
        List of (start, end) regions, sorted by time
    """
    if not segments:
        return []

    texts = [s["text"].strip() for s in segments if s.get("text", "").strip()]
    if not texts:
        return []

    embeddings = _embed(texts)

    # Importance scores: TextRank + optional topic boost
    scores = _compute_textrank_scores(embeddings)
    if topic and topic.strip():
        topic_emb = _embed([topic.strip()])
        topic_sc = _compute_topic_relevance_scores(embeddings, topic_emb)
        scores = 0.6 * scores + 0.4 * (topic_sc / (topic_sc.max() or 1))

    # Top K by score; K ~ target_duration / clip_window
    k = max(1, int(target_duration / clip_window))
    k = min(k, len(segments))

    indices = np.argsort(scores)[::-1][:k]
    selected = [segments[i] for i in indices]

    # Build clip windows centered on each segment
    regions = []
    for seg in selected:
        mid = (seg["start"] + seg["end"]) / 2
        half = clip_window / 2
        start = max(0, mid - half)
        end = mid + half
        regions.append((start, end))

    # Sort by time and merge nearby
    regions.sort(key=lambda r: r[0])
    merged = []
    for start, end in regions:
        if merged and start - merged[-1][1] <= 1.0:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged
