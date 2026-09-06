"""
Hybrid RAG baseline pipeline.

Orchestrates dense + sparse retrieval, fusion, reranking, and Groq generation.
"""
from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from .generation import generate
from .retrievers.dense import retrieve_dense
from .retrievers.fusion import reciprocal_rank_fusion
from .retrievers.reranker import rerank
from .retrievers.sparse import BM25Retriever
from .schema import RAGResult

logger = logging.getLogger("filingsagent.rag")


def query(
    question: str,
    qdrant_client: QdrantClient,
    embed_model: SentenceTransformer,
    bm25_retriever: BM25Retriever,
    reranker_model: CrossEncoder | None = None,
    retrieval_top_k: int = 20,
    final_top_k: int = 5,
    llm_model_name: str | None = None,
) -> RAGResult:
    """
    Full hybrid pipeline: dense + sparse → RRF → rerank → generate.
    """
    dense_results = retrieve_dense(question, qdrant_client, embed_model, top_k=retrieval_top_k)
    logger.info("Dense retrieval: %d results", len(dense_results))

    sparse_results = bm25_retriever.search(question, top_k=retrieval_top_k)
    logger.info("BM25 retrieval: %d results", len(sparse_results))

    fused = reciprocal_rank_fusion(dense_results, sparse_results)[:retrieval_top_k]
    logger.info("RRF fusion: %d candidates", len(fused))

    reranked = rerank(question, fused, top_k=final_top_k, model=reranker_model)
    logger.info("Reranked to top %d", len(reranked))

    answer = generate(question, reranked, model_name=llm_model_name)

    return RAGResult(question=question, answer=answer, retrieved_chunks=reranked)
