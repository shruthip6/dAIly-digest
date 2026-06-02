import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag_embeddings")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embedding_model = None


def _get_embedding_model():
    """Load the embedding model once and reuse it across retrieval operations."""
    global _embedding_model

    if _embedding_model is None:
        try:
            logger.info("Loading embedding model: %s", MODEL_NAME)
            # all-MiniLM-L6-v2 is a lightweight local embedding model optimized for semantic
            # similarity. It runs efficiently on CPU, performs well for common RAG retrieval
            # workloads, has no external API dependency, and avoids embedding usage costs.
            _embedding_model = SentenceTransformer(MODEL_NAME)
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise

    return _embedding_model


def generate_embedding(text: str) -> List[float]:
    """Generate a reusable embedding vector for semantic retrieval."""
    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("Empty text received for embedding generation.")
        return []

    try:
        model = _get_embedding_model()
        embedding = model.encode(text.strip(), normalize_embeddings=True)
        return embedding.tolist()
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        raise
