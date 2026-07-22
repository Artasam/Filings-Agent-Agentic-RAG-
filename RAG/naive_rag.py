"""
Naive RAG baseline: dense retrieval from Qdrant + Gemini generation.

This is deliberately minimal — single dense retriever, top-k chunks, no
reranking, no routing, no self-correction.  It exists to produce the
baseline RAGAS numbers that the full agentic pipeline will be compared
against (blueprint §4: "Baseline vs. improved comparison").

The generation uses Google's Gemini API (free tier) because:
  - Generous free quota (sufficient for eval runs)
  - Strong instruction following for grounded generation
  - Native JSON output support for structured citations later
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .indexer import COLLECTION_NAME, EMBEDDING_MODEL_NAME

logger = logging.getLogger("filingsagent.rag")

TOP_K = 5


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
    """The final output of a naive RAG query."""
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)


SYSTEM_PROMPT = """\
You are a financial analyst assistant.  You answer questions about SEC 10-K \
filings using ONLY the retrieved context provided below.  If the context does \
not contain enough information to answer the question, say "I don't have \
sufficient information in the retrieved filings to answer this question."

Rules:
- Base every claim strictly on the provided context.
- Cite the source chunk when making a specific claim (use the chunk_id).
- Do not speculate or use information not present in the context.
- Be concise and precise.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Formats retrieved chunks into a context block for the LLM prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        header = (
            f"[Chunk {i}] chunk_id={c.chunk_id[:12]}... | "
            f"ticker={c.ticker} | section={c.section_name} | "
            f"filing_date={c.filing_date}"
        )
        parts.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def retrieve(
    query: str,
    client: QdrantClient,
    model: SentenceTransformer,
    top_k: int = TOP_K,
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


def generate(
    question: str,
    chunks: list[RetrievedChunk],
    model_name: str = "gemini-2.0-flash",
) -> str:
    """Calls Gemini to generate an answer grounded in retrieved chunks."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)
    context = _format_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    return response.text


def query(
    question: str,
    qdrant_client: QdrantClient,
    embed_model: SentenceTransformer,
    top_k: int = TOP_K,
    llm_model_name: str = "gemini-2.0-flash",
) -> RAGResult:
    """End-to-end naive RAG: retrieve chunks then generate an answer."""
    chunks = retrieve(question, qdrant_client, embed_model, top_k=top_k)
    answer = generate(question, chunks, model_name=llm_model_name)
    return RAGResult(question=question, answer=answer, retrieved_chunks=chunks)
