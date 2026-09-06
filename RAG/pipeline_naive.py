"""
Naive RAG baseline pipeline.

Orchestrates dense retrieval and Groq generation.
"""
from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from RAG.hf_api_embedder import HFInferenceEmbedder

from .generation import generate
from .retrievers.dense import retrieve_dense
from .schema import RAGResult

logger = logging.getLogger("filingsagent.rag")


def query(
    question: str,
    qdrant_client: QdrantClient,
    embed_model: HFInferenceEmbedder,
    top_k: int = 5,
    llm_model_name: str | None = None,
) -> RAGResult:
    """End-to-end naive RAG: retrieve chunks then generate an answer."""
    chunks = retrieve_dense(question, qdrant_client, embed_model, top_k=top_k)
    answer = generate(question, chunks, model_name=llm_model_name)
    return RAGResult(question=question, answer=answer, retrieved_chunks=chunks)
