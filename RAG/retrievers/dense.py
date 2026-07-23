"""
Dense vector retriever using Qdrant.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ..indexer import COLLECTION_NAME
from ..schema import RetrievedChunk


def retrieve_dense(
    query: str,
    client: QdrantClient,
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Dense vector search against the Qdrant filing_chunks collection."""
    # nomic-embed-text-v1.5 requires 'search_query:' prefix for queries
    query_vector = model.encode("search_query: " + query).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    chunks = []
    for point in results.points:
        p = point.payload
        chunks.append(
            RetrievedChunk(
                chunk_id=p.get("chunk_id", ""),
                text=p.get("text", ""),
                score=point.score,
                ticker=p.get("ticker", ""),
                section_name=p.get("section_name", ""),
                accession_no=p.get("accession_no", ""),
                filing_date=p.get("filing_date", ""),
                fiscal_year=p.get("fiscal_year"),
            )
        )
    return chunks
