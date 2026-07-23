"""
Naive RAG baseline pipeline.

Orchestrates dense retrieval and Gemini generation.
"""
from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .generation import generate
from .retrievers.dense import retrieve_dense
from .schema import RAGResult

logger = logging.getLogger("filingsagent.rag")


def query(
    question: str,
    qdrant_client: QdrantClient,
    embed_model: SentenceTransformer,
    top_k: int = 5,
    llm_model_name: str = "gemini-2.0-flash",
) -> RAGResult:
    """End-to-end naive RAG: retrieve chunks then generate an answer."""
    chunks = retrieve_dense(question, qdrant_client, embed_model, top_k=top_k)
    answer = generate(question, chunks, model_name=llm_model_name)
    return RAGResult(question=question, answer=answer, retrieved_chunks=chunks)
