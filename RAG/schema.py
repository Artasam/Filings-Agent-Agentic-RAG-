"""
Core data structures for the RAG pipelines.
"""
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """A single chunk returned by retrieval, with its metadata and score."""
    chunk_id: str
    text: str
    score: float
    ticker: str = ""
    section_name: str = ""
    accession_no: str = ""
    filing_date: str = ""
    fiscal_year: int | None = None


@dataclass
class RAGResult:
    """The final output of a RAG query."""
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
