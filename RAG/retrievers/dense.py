"""
Dense vector retriever using Qdrant.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from RAG.hf_api_embedder import HFInferenceEmbedder
from ..indexer import COLLECTION_NAME
from ..schema import RetrievedChunk


def retrieve_dense(
    query: str,
    client: QdrantClient,
    model: HFInferenceEmbedder,
    top_k: int = 5,
    filters: dict | None = None,
) -> list[RetrievedChunk]:
    """Dense vector search against the Qdrant filing_chunks collection."""
    # BAAI/bge-m3 does not require instruction prefixes — send raw query text.
    query_vector = model.encode(query).tolist()[0]

    qdrant_filter = None
    if filters:
        qdrant_filter = Filter(
            must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
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
