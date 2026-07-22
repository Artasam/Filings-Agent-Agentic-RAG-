"""
Offline tests for Phase 2: RAG indexer and naive retrieval.

These use in-memory Qdrant and a tiny mock SentenceTransformer so they
run with zero network access and no GPU.  They verify the wiring:
  - Chunks flow from SQLite -> embeddings -> Qdrant correctly
  - Dense retrieval returns expected results with correct metadata
  - Re-indexing is idempotent (same count, no duplicates)

Run with:  python -m pytest tests/test_rag_baseline.py -v
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from Ingestion.storage import Storage, make_id
from RAG.indexer import COLLECTION_NAME, build_index
from RAG.naive_rag import RetrievedChunk, _format_context, retrieve


def _seed_db(db_path: Path) -> int:
    """Inserts a small set of realistic chunks into a fresh SQLite DB."""
    storage = Storage(db_path)
    storage.upsert_company(cik="0000320193", ticker="AAPL")
    storage.upsert_filing(
        accession_no="0000320193-24-000001",
        cik="0000320193",
        form="10-K",
        filing_date="2024-11-01",
        fiscal_year=2024,
        fiscal_period="FY",
        status="chunked",
    )

    texts = [
        "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables.",
        "The company faces significant competition in all markets where it operates.",
        "Revenue increased year over year driven by strong iPhone and Services demand.",
        "Research and development expenses were driven by ongoing investment in new products and technologies.",
        "The company is exposed to foreign currency exchange rate fluctuations.",
    ]
    rows = []
    for i, text in enumerate(texts):
        chunk_id = make_id("0000320193-24-000001", "item_1a_risk_factors", str(i), text[:64])
        rows.append((
            chunk_id, "0000320193-24-000001", "0000320193",
            "item_1a_risk_factors", i, len(text.split()), text,
        ))
    storage.upsert_chunks(rows)
    storage.close()
    return len(rows)


class FakeEmbedder:
    """
    Deterministic mock that maps text to a fixed-dim vector based on a
    hash, so 'similar' texts (same text) always get the same vector and
    retrieval ordering is reproducible.

    Mirrors SentenceTransformer.encode() behavior: single string → 1D
    array, list of strings → 2D array.
    """
    def encode(self, texts, show_progress_bar=False):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        vectors = []
        for t in texts:
            # Deterministic pseudo-random vector from text hash
            rng = np.random.RandomState(hash(t) % (2**31))
            vectors.append(rng.randn(768).astype(np.float32))
        result = np.array(vectors)
        return result[0] if single else result


def test_build_index_populates_qdrant():
    """Chunks in SQLite → embeddings → Qdrant points."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        n_chunks = _seed_db(db_path)

        client = QdrantClient(":memory:")
        model = FakeEmbedder()

        count = build_index(db_path, client=client, model=model)
        assert count == n_chunks

        info = client.get_collection(COLLECTION_NAME)
        assert info.points_count == n_chunks


def test_build_index_is_idempotent():
    """Re-indexing replaces the collection cleanly, no duplicates."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        n_chunks = _seed_db(db_path)

        client = QdrantClient(":memory:")
        model = FakeEmbedder()

        build_index(db_path, client=client, model=model)
        build_index(db_path, client=client, model=model)

        info = client.get_collection(COLLECTION_NAME)
        assert info.points_count == n_chunks


def test_retrieve_returns_top_k_with_metadata():
    """Dense search returns the right number of results with correct payload fields."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        _seed_db(db_path)

        client = QdrantClient(":memory:")
        model = FakeEmbedder()
        build_index(db_path, client=client, model=model)

        results = retrieve("competition risk", client, model, top_k=3)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, RetrievedChunk)
            assert r.chunk_id  # not empty
            assert r.text  # not empty
            assert r.ticker == "AAPL"
            assert r.section_name == "item_1a_risk_factors"
            assert r.score is not None


def test_build_index_returns_zero_on_empty_db():
    """No chunks in DB → 0 indexed, no crash."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        storage = Storage(db_path)
        storage.close()

        client = QdrantClient(":memory:")
        model = FakeEmbedder()
        count = build_index(db_path, client=client, model=model)
        assert count == 0


def test_format_context_produces_readable_output():
    """Sanity check that the LLM context string is well-formed."""
    chunks = [
        RetrievedChunk(
            chunk_id="abc123def456", text="Revenue grew 10%.", score=0.92,
            ticker="AAPL", section_name="item_7_mdna",
            accession_no="0000320193-24-000001", filing_date="2024-11-01",
        ),
    ]
    ctx = _format_context(chunks)
    assert "Revenue grew 10%." in ctx
    assert "ticker=AAPL" in ctx
    assert "item_7_mdna" in ctx


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
