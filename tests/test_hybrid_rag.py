"""
Offline tests for Phase 3: BM25 retriever, RRF fusion, reranker, and
hybrid retrieval pipeline.

All tests use in-memory Qdrant, a FakeEmbedder (no model download), and
a FakeReranker (deterministic scoring) so they run in seconds with zero
network access.

Run with:  python -m pytest tests/test_hybrid_rag.py -v
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient

from Ingestion.storage import Storage, make_id
from RAG.indexer import build_index
from RAG.retrievers.dense import retrieve_dense as retrieve
from RAG.retrievers.fusion import reciprocal_rank_fusion
from RAG.retrievers.reranker import rerank
from RAG.retrievers.sparse import BM25Retriever
from RAG.schema import RetrievedChunk


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {"chunk_id": "aaa", "text": "Apple faces significant competition in smartphones and tablets.",
     "ticker": "AAPL", "cik": "0000320193", "accession_no": "acc-1",
     "section_name": "item_1a_risk_factors", "chunk_index": 0,
     "filing_date": "2024-11-01", "fiscal_year": 2024, "form": "10-K",
     "fiscal_period": "FY", "token_count": 9},
    {"chunk_id": "bbb", "text": "Revenue increased due to strong iPhone and Services demand.",
     "ticker": "AAPL", "cik": "0000320193", "accession_no": "acc-1",
     "section_name": "item_7_mdna", "chunk_index": 0,
     "filing_date": "2024-11-01", "fiscal_year": 2024, "form": "10-K",
     "fiscal_period": "FY", "token_count": 9},
    {"chunk_id": "ccc", "text": "Research and development expenses grew as the company invested in new products.",
     "ticker": "AAPL", "cik": "0000320193", "accession_no": "acc-1",
     "section_name": "item_7_mdna", "chunk_index": 1,
     "filing_date": "2024-11-01", "fiscal_year": 2024, "form": "10-K",
     "fiscal_period": "FY", "token_count": 12},
    {"chunk_id": "ddd", "text": "Microsoft Azure cloud revenue continued to grow rapidly in fiscal 2024.",
     "ticker": "MSFT", "cik": "0000789019", "accession_no": "acc-2",
     "section_name": "item_7_mdna", "chunk_index": 0,
     "filing_date": "2024-10-01", "fiscal_year": 2024, "form": "10-K",
     "fiscal_period": "FY", "token_count": 11},
    {"chunk_id": "eee", "text": "The company is exposed to foreign currency exchange rate fluctuations.",
     "ticker": "AAPL", "cik": "0000320193", "accession_no": "acc-1",
     "section_name": "item_1a_risk_factors", "chunk_index": 1,
     "filing_date": "2024-11-01", "fiscal_year": 2024, "form": "10-K",
     "fiscal_period": "FY", "token_count": 10},
]


def _seed_db(db_path: Path) -> int:
    """Inserts SAMPLE_CHUNKS into a fresh SQLite DB."""
    storage = Storage(db_path)
    storage.upsert_company(cik="0000320193", ticker="AAPL")
    storage.upsert_company(cik="0000789019", ticker="MSFT")
    storage.upsert_filing(
        accession_no="acc-1", cik="0000320193", form="10-K",
        filing_date="2024-11-01", fiscal_year=2024, fiscal_period="FY",
        status="chunked",
    )
    storage.upsert_filing(
        accession_no="acc-2", cik="0000789019", form="10-K",
        filing_date="2024-10-01", fiscal_year=2024, fiscal_period="FY",
        status="chunked",
    )
    rows = [
        (c["chunk_id"], c["accession_no"], c["cik"],
         c["section_name"], c["chunk_index"], c["token_count"], c["text"])
        for c in SAMPLE_CHUNKS
    ]
    storage.upsert_chunks(rows)
    storage.close()
    return len(rows)


class FakeEmbedder:
    """Deterministic embedder (same as test_rag_baseline.py)."""
    def encode(self, texts, show_progress_bar=False):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        vectors = []
        for t in texts:
            rng = np.random.RandomState(hash(t) % (2**31))
            vectors.append(rng.randn(768).astype(np.float32))
        result = np.array(vectors)
        return result[0] if single else result


class FakeReranker:
    """
    Mock cross-encoder that scores each (query, text) pair by the number
    of shared words.  Deterministic, no model download needed.
    """
    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            q_words = set(query.lower().split())
            t_words = set(text.lower().split())
            scores.append(float(len(q_words & t_words)))
        return np.array(scores)


# ---------------------------------------------------------------------------
# Tests: BM25 Retriever
# ---------------------------------------------------------------------------

def test_bm25_returns_results_with_metadata():
    """BM25 search returns chunks with correct metadata fields."""
    retriever = BM25Retriever(SAMPLE_CHUNKS)
    results = retriever.search("competition smartphones", top_k=3)
    assert len(results) >= 1
    top = results[0]
    assert isinstance(top, RetrievedChunk)
    assert "competition" in top.text.lower()
    assert top.ticker == "AAPL"
    assert top.score > 0


def test_bm25_keyword_matching_beats_semantic():
    """BM25 should rank 'Azure cloud' higher for Azure-specific query."""
    retriever = BM25Retriever(SAMPLE_CHUNKS)
    results = retriever.search("Azure cloud revenue", top_k=5)
    assert results[0].chunk_id == "ddd"  # the Azure chunk


def test_bm25_empty_index_returns_empty():
    retriever = BM25Retriever([])
    assert retriever.search("anything") == []


# ---------------------------------------------------------------------------
# Tests: RRF Fusion
# ---------------------------------------------------------------------------

def _make_chunk(cid: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, text=f"text_{cid}", score=score)


def test_rrf_merges_two_lists():
    """RRF produces a single deduped ranked list from two inputs."""
    list_a = [_make_chunk("a", 0.9), _make_chunk("b", 0.8), _make_chunk("c", 0.7)]
    list_b = [_make_chunk("b", 5.0), _make_chunk("d", 4.0), _make_chunk("a", 3.0)]
    fused = reciprocal_rank_fusion(list_a, list_b, k=60)

    ids = [c.chunk_id for c in fused]
    assert len(ids) == len(set(ids)), "RRF output should have no duplicates"
    assert len(ids) == 4  # a, b, c, d

    # "b" is rank 2 in list_a and rank 1 in list_b → highest combined RRF
    # "a" is rank 1 in list_a and rank 3 in list_b → second highest
    assert ids[0] == "b"
    assert ids[1] == "a"


def test_rrf_single_list_preserves_order():
    """With one list, RRF should preserve the original ranking."""
    items = [_make_chunk("x", 0.9), _make_chunk("y", 0.5), _make_chunk("z", 0.1)]
    fused = reciprocal_rank_fusion(items, k=60)
    assert [c.chunk_id for c in fused] == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Tests: Reranker
# ---------------------------------------------------------------------------

def test_reranker_reorders_by_cross_encoder_score():
    """FakeReranker scores by word overlap, so it should reorder chunks."""
    chunks = [
        _make_chunk("low_overlap", 0.9),   # original score high but text has few shared words
        _make_chunk("high_overlap", 0.1),   # original score low but text will match
    ]
    # Override texts for the test
    chunks[0].text = "foreign currency exchange rate fluctuations"
    chunks[1].text = "competition in smartphones and competition risk factors"

    result = rerank("competition risk", chunks, top_k=2, model=FakeReranker())
    # "competition risk" shares 2 words with chunk[1] and 0 with chunk[0]
    assert result[0].chunk_id == "high_overlap"
    assert result[0].score > result[1].score


def test_reranker_empty_input():
    assert rerank("anything", [], top_k=5, model=FakeReranker()) == []


# ---------------------------------------------------------------------------
# Tests: End-to-end hybrid retrieval (without generation)
# ---------------------------------------------------------------------------

def test_hybrid_retrieval_combines_dense_and_sparse():
    """Dense + BM25 + RRF + rerank produces final_top_k results."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        _seed_db(db_path)

        # Build Qdrant index
        qdrant = QdrantClient(":memory:")
        embedder = FakeEmbedder()
        build_index(db_path, client=qdrant, model=embedder)

        # Build BM25 index
        bm25 = BM25Retriever(SAMPLE_CHUNKS)

        # Dense retrieval
        dense = retrieve("competition risk", qdrant, embedder, top_k=5)
        # Sparse retrieval
        sparse = bm25.search("competition risk", top_k=5)

        assert len(dense) > 0
        assert len(sparse) > 0

        # RRF fusion
        fused = reciprocal_rank_fusion(dense, sparse)
        assert len(fused) >= 1
        # No duplicates
        fused_ids = [c.chunk_id for c in fused]
        assert len(fused_ids) == len(set(fused_ids))

        # Rerank
        reranked = rerank("competition risk", fused[:5], top_k=3, model=FakeReranker())
        assert len(reranked) == 3
        # Scores should be from the reranker, not from retrieval
        for r in reranked:
            assert r.score is not None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
