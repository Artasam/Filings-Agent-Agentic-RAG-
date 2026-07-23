"""
BM25 sparse retriever over ingested filing chunks.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ..indexer import _load_chunks_from_db
from ..schema import RetrievedChunk

logger = logging.getLogger("filingsagent.rag")


class BM25Retriever:
    """In-memory BM25 index over chunk texts."""

    def __init__(self, chunks: list[dict]):
        if not chunks:
            self._chunks: list[dict] = []
            self._index = None
            return

        self._chunks = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self._index = BM25Okapi(tokenized)

    @classmethod
    def from_db(cls, db_path: Path) -> "BM25Retriever":
        chunks = _load_chunks_from_db(db_path)
        return cls(chunks)

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        if self._index is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            c = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    score=float(scores[idx]),
                    ticker=c.get("ticker", ""),
                    section_name=c.get("section_name", ""),
                    accession_no=c.get("accession_no", ""),
                    filing_date=c.get("filing_date", ""),
                    fiscal_year=c.get("fiscal_year"),
                )
            )
        return results
