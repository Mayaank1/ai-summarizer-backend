"""
Embedding-based key moment detection.
Uses sentence embeddings + TextRank for semantic importance scoring.
"""
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import Config
from logger import get_logger

logger = get_logger()

# Lazy-loaded model
_model = None


def _get_embedding_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = getattr(Config, "EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            _model = SentenceTransformer(model_name)
            logger.info("Loaded embedding model: %s", model_name)
        except ImportError as e:
            logger.error("sentence-transformers not installed: %s", e)
            raise
    return _model


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

    model = _get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)

    # Importance scores: TextRank + optional topic boost
    scores = _compute_textrank_scores(embeddings)
    if topic and topic.strip():
        topic_emb = model.encode([topic.strip()], show_progress_bar=False)
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
