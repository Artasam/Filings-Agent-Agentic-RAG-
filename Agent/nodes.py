"""
Node functions for the agentic RAG graph.

Each function takes the current AgentState and any injected resources,
performs its task, and returns a partial state update dict.  Resources
(LLM client, Qdrant, models) are bound via closures in graph.py — this
keeps each function pure and independently testable.

Structured outputs use Pydantic schemas + response_schema in the Gemini
API config, which enforces the shape at the API level rather than just
asking nicely with a prompt. This is far more reliable than
response_mime_type="application/json" + json.loads().
"""
from __future__ import annotations

import logging
from typing import Literal

from google import genai
from pydantic import BaseModel, Field

from Ingestion.config import XBRL_CONCEPTS
from Ingestion.storage import Storage
from RAG.retrievers.dense import retrieve_dense
from RAG.retrievers.fusion import reciprocal_rank_fusion
from RAG.retrievers.reranker import rerank
from RAG.retrievers.sparse import BM25Retriever

logger = logging.getLogger("filingsagent.agent")

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM outputs
# ---------------------------------------------------------------------------

class RouteDecision(BaseModel):
    """Schema for the router node output."""
    route: Literal["vector_search", "sql_search", "out_of_scope"] = Field(
        description="The retrieval path to take for this query."
    )


class SQLParams(BaseModel):
    """Schema for the SQL parameter extraction node output."""
    ticker: str | None = Field(
        description="Company ticker symbol in uppercase (e.g. AAPL), or null if not mentioned."
    )
    concept: str | None = Field(
        description="The closest XBRL us-gaap concept name (e.g. NetIncomeLoss), or null."
    )
    fiscal_year: int | None = Field(
        description="The fiscal year as a 4-digit integer (e.g. 2024), or null."
    )


class GraderDecision(BaseModel):
    """Schema for the grader node output."""
    passed: bool = Field(
        description="True if the answer is grounded in the context, False if it is insufficient or hallucinated."
    )


# ---------------------------------------------------------------------------
# Prompts  (no longer need to say "respond with ONLY a JSON object" —
# the schema enforces that at the API level)
# ---------------------------------------------------------------------------

ROUTER_PROMPT = """\
You are a query router for a financial analysis system with access to
SEC 10-K filing documents and structured XBRL financial data.

Classify the user's question into exactly one of:
- "vector_search": qualitative questions about filings (risk factors,
  business strategy, management discussion, competitive landscape).
- "sql_search": questions asking for a specific financial number
  (revenue, net income, EPS, assets, R&D expense, etc.) for a named
  company and/or time period.
- "out_of_scope": questions unrelated to SEC filings or financial analysis.

Question: {question}"""

SQL_EXTRACT_PROMPT = """\
Extract the structured parameters from the following financial question.

Available XBRL concepts (pick the closest match or null):
{concepts}

Question: {question}"""

GENERATE_PROMPT = """\
You are a financial analyst assistant. Answer the question using ONLY
the context provided below.  If the context is insufficient, say
"I don't have sufficient information in the retrieved filings to answer
this question."

Rules:
- Base every claim strictly on the provided context.
- Cite sources when making specific claims.
- Do not speculate beyond the context.
- Be concise and precise.

{context}

Question: {question}"""

GRADER_PROMPT = """\
You are a grading assistant. Evaluate whether the answer adequately
addresses the question using the provided context.

Question: {question}
Answer: {answer}

Does the answer directly address the question with information from
the context (not fabricated)?"""


# ---------------------------------------------------------------------------
# Helper: single place to call Gemini with a response_schema
# ---------------------------------------------------------------------------

def _structured_call(client: genai.Client, prompt: str, schema: type[BaseModel]) -> BaseModel:
    """
    Calls Gemini with a Pydantic schema enforced at the API level via
    response_schema.  This guarantees the output matches the schema shape,
    unlike response_mime_type="application/json" which only hints at it.
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return schema.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def route_question(state: dict, client: genai.Client) -> dict:
    """Classifies the question into vector_search, sql_search, or out_of_scope."""
    prompt = ROUTER_PROMPT.format(question=state["question"])
    decision: RouteDecision = _structured_call(client, prompt, RouteDecision)
    logger.info("Router decided: %s", decision.route)
    return {"route": decision.route}


def vector_search(state: dict, qdrant_client, embed_model, bm25, reranker) -> dict:
    """Hybrid retrieval: dense + BM25 → RRF → rerank → top 5."""
    question = state["question"]
    dense = retrieve_dense(question, qdrant_client, embed_model, top_k=20)
    sparse = bm25.search(question, top_k=20)
    fused = reciprocal_rank_fusion(dense, sparse)[:20]
    reranked = rerank(question, fused, top_k=5, model=reranker)
    # Convert to dicts for JSON-serialisable state
    docs = [
        {"chunk_id": c.chunk_id, "text": c.text, "score": c.score,
         "ticker": c.ticker, "section_name": c.section_name,
         "accession_no": c.accession_no, "filing_date": c.filing_date,
         "fiscal_year": c.fiscal_year}
        for c in reranked
    ]
    logger.info("Vector search returned %d chunks", len(docs))
    return {"documents": docs}


def sql_search(state: dict, client: genai.Client, storage: Storage) -> dict:
    """Extracts query params via LLM (structured output), then looks up XBRL facts in SQLite."""
    concepts_str = ", ".join(XBRL_CONCEPTS)
    prompt = SQL_EXTRACT_PROMPT.format(
        question=state["question"], concepts=concepts_str,
    )
    params: SQLParams = _structured_call(client, prompt, SQLParams)

    if not params.ticker:
        logger.warning("SQL search: no ticker extracted from question")
        return {"sql_results": []}

    cik = storage.get_cik_for_ticker(params.ticker)
    if not cik:
        logger.warning("SQL search: ticker %s not found in DB", params.ticker)
        return {"sql_results": []}

    facts = storage.get_xbrl_facts(cik, concept=params.concept, fiscal_year=params.fiscal_year)
    logger.info("SQL search: %d facts for %s/%s/%s", len(facts), params.ticker, params.concept, params.fiscal_year)
    return {"sql_results": facts}


def generate_answer(state: dict, client: genai.Client) -> dict:
    """Generates an answer grounded in retrieved documents and/or SQL results."""
    context_parts = []

    # Vector-retrieved chunks
    for i, doc in enumerate(state.get("documents") or [], 1):
        header = (
            f"[Chunk {i}] ticker={doc.get('ticker','')} | "
            f"section={doc.get('section_name','')} | "
            f"filing_date={doc.get('filing_date','')}"
        )
        context_parts.append(f"{header}\n{doc['text']}")

    # SQL results
    for fact in state.get("sql_results") or []:
        context_parts.append(
            f"[XBRL Fact] concept={fact['concept']} | "
            f"value={fact['value']} {fact.get('unit','')} | "
            f"fiscal_year={fact.get('fiscal_year','')} | "
            f"period={fact.get('fiscal_period','')}"
        )

    if not context_parts:
        return {"generation": "I don't have sufficient information in the retrieved filings to answer this question."}

    context = "\n\n---\n\n".join(context_parts)
    prompt = GENERATE_PROMPT.format(context=f"Context:\n{context}", question=state["question"])

    # Generation is free-form text, no structured schema needed here
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return {"generation": response.text}


def grade_answer(state: dict, client: genai.Client) -> dict:
    """
    Checks if the generation adequately answers the question.
    Returns the updated routing state. The edge decision (pass vs retry)
    is handled by the conditional edge in graph.py.
    """
    prompt = GRADER_PROMPT.format(
        question=state["question"], answer=state.get("generation", ""),
    )
    decision: GraderDecision = _structured_call(client, prompt, GraderDecision)
    retries = state.get("retries", 0)

    if decision.passed:
        logger.info("Grader: PASS")
        return {"route": "pass"}
    elif retries >= MAX_RETRIES:
        logger.warning("Grader: FAIL — retry cap (%d) reached", MAX_RETRIES)
        return {
            "route": "fail_cap",
            "generation": "I don't have sufficient information in the retrieved filings to answer this question.",
        }
    else:
        logger.info("Grader: FAIL — retrying (%d/%d)", retries + 1, MAX_RETRIES)
        return {"route": "retry", "retries": retries + 1}
