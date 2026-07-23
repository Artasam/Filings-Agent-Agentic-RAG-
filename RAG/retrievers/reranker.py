"""
Cross-encoder reranker.
"""
from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from ..schema import RetrievedChunk

logger = logging.getLogger("filingsagent.rag")

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
    model: CrossEncoder | None = None,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    if model is None:
        logger.info("Loading reranker model: %s", RERANKER_MODEL_NAME)
        model = CrossEncoder(RERANKER_MODEL_NAME)

    pairs = [[query, c.text] for c in chunks]
    scores = model.predict(pairs)

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    results = []
    for score, chunk in scored[:top_k]:
        results.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(score),
                ticker=chunk.ticker,
                section_name=chunk.section_name,
                accession_no=chunk.accession_no,
                filing_date=chunk.filing_date,
                fiscal_year=chunk.fiscal_year,
            )
        )
    return results
