"""
Embeds ingested chunks and indexes them into a Qdrant vector store.

This is the bridge between Phase 1 (ingestion into SQLite) and Phase 2
(retrieval).  It reads every chunk from the `chunks` table, encodes it
with the chosen embedding model, and upserts the vectors + metadata into
a Qdrant collection.  Re-running is idempotent: points are keyed by
chunk_id, so duplicates are silently overwritten with the same data.

Embedding model choice (see implementation_plan.md for full rationale):
  nomic-ai/nomic-embed-text-v1.5
  - 8192-token context window (our chunks are ~650 tokens — no truncation)
  - Open weights, self-hostable, no API cost
  - Outperforms OpenAI ada-002 / text-embedding-3-small on MTEB benchmarks
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from Ingestion.storage import Storage

logger = logging.getLogger("filingsagent.rag")

EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
COLLECTION_NAME = "filing_chunks"
EMBEDDING_DIM = 768  # nomic-embed-text-v1.5 output dimension
BATCH_SIZE = 64       # encode+upsert this many chunks at a time


def _load_chunks_from_db(db_path: Path) -> list[dict]:
    """Reads all chunks from the ingestion SQLite database."""
    storage = Storage(db_path)
    try:
        with storage.cursor() as cur:
            cur.execute(
                "SELECT c.chunk_id, c.accession_no, c.cik, c.section_name, "
                "c.chunk_index, c.token_count, c.text, "
                "f.form, f.filing_date, f.fiscal_year, f.fiscal_period, "
                "co.ticker "
                "FROM chunks c "
                "JOIN filings f ON c.accession_no = f.accession_no "
                "JOIN companies co ON c.cik = co.cik"
            )
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        storage.close()


def build_index(
    db_path: Path,
    qdrant_path: str | None = None,
    model: SentenceTransformer | None = None,
    client: QdrantClient | None = None,
) -> int:
    """
    Reads chunks from SQLite, embeds them, and upserts into Qdrant.

    Args:
        db_path: Path to the ingestion SQLite database.
        qdrant_path: Directory for Qdrant on-disk storage. If None, uses
                     in-memory storage (useful for tests).
        model: Pre-loaded SentenceTransformer. If None, loads the default.
        client: Pre-built QdrantClient. If None, creates one from qdrant_path.

    Returns:
        Number of points indexed.
    """
    chunks = _load_chunks_from_db(db_path)
    if not chunks:
        logger.warning("No chunks found in database — nothing to index.")
        return 0

    if model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)

    if client is None:
        if qdrant_path:
            client = QdrantClient(path=qdrant_path)
        else:
            client = QdrantClient(":memory:")

    # Recreate the collection to ensure a clean state on re-index.
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    total_indexed = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        # nomic-embed-text-v1.5 requires a task-type prefix for best results.
        texts = ["search_document: " + c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False)

        points = []
        for j, chunk in enumerate(batch):
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
                    vector=embeddings[j].tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "ticker": chunk["ticker"],
                        "cik": chunk["cik"],
                        "accession_no": chunk["accession_no"],
                        "section_name": chunk["section_name"],
                        "chunk_index": chunk["chunk_index"],
                        "form": chunk["form"],
                        "filing_date": chunk["filing_date"],
                        "fiscal_year": chunk["fiscal_year"],
                        "fiscal_period": chunk["fiscal_period"],
                        "text": chunk["text"],
                    },
                )
            )
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_indexed += len(points)
        logger.info("Indexed %d / %d chunks", total_indexed, len(chunks))

    logger.info("Indexing complete: %d points in collection '%s'", total_indexed, COLLECTION_NAME)
    return total_indexed
