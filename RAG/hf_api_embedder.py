"""
Local SentenceTransformer Embedder using BAAI/bge-m3.

Uses the locally cached model (already downloaded in HuggingFace cache)
instead of making API calls. This eliminates 402 Payment Required errors.

The .encode() interface is identical to the old HFInferenceEmbedder
so all callers (indexer, dense retriever, agent, run_eval) need zero changes.
"""
import logging
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load .env so GOOGLE_API_KEY / HF_TOKEN are available to other components
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger("filingsagent.rag.embedder")

# Lazy singleton — model is loaded once and reused across all calls
_MODEL_INSTANCE: SentenceTransformer | None = None


def _get_model(model_name: str) -> SentenceTransformer:
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        logger.info("Loading local model: %s (first call only)", model_name)
        _MODEL_INSTANCE = SentenceTransformer(model_name, trust_remote_code=True)
        logger.info("Model ready.")
    return _MODEL_INSTANCE


class HFInferenceEmbedder:
    """
    Local embedding using SentenceTransformer (BAAI/bge-m3).
    Named HFInferenceEmbedder for backward compatibility — no callers need changing.
    Uses locally cached HuggingFace weights — zero API credits consumed.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.model = _get_model(model_name)
        logger.info("HFInferenceEmbedder (local) ready: %s", model_name)

    def encode(self, texts: list[str] | str, batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Embed texts locally. Returns ndarray of shape (len(texts), 1024).
        normalize_embeddings=True is BGE-M3 best practice for cosine similarity.
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings)
