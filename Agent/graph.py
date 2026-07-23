"""
Builds and compiles the LangGraph agent.

The graph wires the node functions from nodes.py into a state machine
with conditional edges for routing (vector vs SQL vs out-of-scope) and
self-correction (grade → retry loop with a hard cap).
"""
from __future__ import annotations

import os
from pathlib import Path

from google import genai
from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from Ingestion.storage import Storage
from RAG.indexer import EMBEDDING_MODEL_NAME
from RAG.retrievers.reranker import RERANKER_MODEL_NAME
from RAG.retrievers.sparse import BM25Retriever

from . import nodes
from .state import AgentState


def _route_after_router(state: AgentState) -> str:
    """Conditional edge after the router node."""
    route = state.get("route", "vector_search")
    if route == "out_of_scope":
        return "out_of_scope"
    elif route == "sql_search":
        return "sql_search"
    return "vector_search"


def _route_after_grader(state: AgentState) -> str:
    """Conditional edge after the grader node."""
    route = state.get("route", "pass")
    if route == "retry":
        return "vector_search"
    return END


def build_agent(
    db_path: Path,
    qdrant_client: QdrantClient | None = None,
    embed_model: SentenceTransformer | None = None,
    bm25: BM25Retriever | None = None,
    reranker: CrossEncoder | None = None,
    api_key: str | None = None,
) -> StateGraph:
    """
    Constructs and compiles the agentic RAG graph.

    All heavy resources (models, clients) are loaded once here and
    captured by closures so node functions stay pure.
    """
    api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GOOGLE_API_KEY or GEMINI_API_KEY")

    llm_client = genai.Client(api_key=api_key)
    storage = Storage(db_path)

    if qdrant_client is None:
        qdrant_client = QdrantClient(path=str(Path("data") / "qdrant_store"))
    if embed_model is None:
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    if bm25 is None:
        bm25 = BM25Retriever.from_db(db_path)
    if reranker is None:
        reranker = CrossEncoder(RERANKER_MODEL_NAME)

    # --- Bind resources to node functions via closures ---
    def router(state):
        return nodes.route_question(state, llm_client)

    def vector_search(state):
        return nodes.vector_search(state, qdrant_client, embed_model, bm25, reranker)

    def sql_search(state):
        return nodes.sql_search(state, llm_client, storage)

    def generate(state):
        return nodes.generate_answer(state, llm_client)

    def grade(state):
        return nodes.grade_answer(state, llm_client)

    def out_of_scope(state):
        return {"generation": "This question is outside the scope of SEC financial filings analysis."}

    # --- Wire the graph ---
    graph = StateGraph(AgentState)

    graph.add_node("router", router)
    graph.add_node("vector_search", vector_search)
    graph.add_node("sql_search", sql_search)
    graph.add_node("generate", generate)
    graph.add_node("grade", grade)
    graph.add_node("out_of_scope", out_of_scope)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", _route_after_router, {
        "vector_search": "vector_search",
        "sql_search": "sql_search",
        "out_of_scope": "out_of_scope",
    })

    graph.add_edge("vector_search", "generate")
    graph.add_edge("sql_search", "generate")
    graph.add_edge("out_of_scope", END)
    graph.add_edge("generate", "grade")

    graph.add_conditional_edges("grade", _route_after_grader, {
        "vector_search": "vector_search",
        END: END,
    })

    return graph.compile()
