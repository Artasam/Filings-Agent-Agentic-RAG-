"""
Builds and compiles the LangGraph agent.

Phase 5 graph flow:
  analyze_query → [out_of_scope → save_turn → END]
                → execute_retrieval → generate → verify_citations
                → [pass → save_turn → END]
                → [retry → execute_retrieval (loop, capped)]
                → [fail_cap → save_turn → END]

MemorySaver checkpointer enables multi-turn conversations by persisting
the messages list across invocations (keyed by thread_id in config).
"""
from __future__ import annotations

import os
from pathlib import Path

from groq import Groq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sentence_transformers import CrossEncoder

from RAG.hf_api_embedder import HFInferenceEmbedder

from Ingestion.storage import Storage
from RAG.indexer import EMBEDDING_MODEL_NAME
from RAG.retrievers.reranker import RERANKER_MODEL_NAME
from RAG.retrievers.sparse import BM25Retriever

from . import nodes
from .state import AgentState


def _route_after_analyzer(state: AgentState) -> str:
    """Conditional edge after the analyzer node."""
    if state.get("route") == "out_of_scope":
        return "out_of_scope"
    return "execute_retrieval"


def _route_after_verifier(state: AgentState) -> str:
    """Conditional edge after the verifier node."""
    route = state.get("route", "pass")
    if route == "retry":
        return "execute_retrieval"
    return "save_turn"


def build_agent(
    db_path: Path,
    qdrant_client: QdrantClient | None = None,
    embed_model: HFInferenceEmbedder | None = None,
    bm25: BM25Retriever | None = None,
    reranker: CrossEncoder | None = None,
    api_key: str | None = None,
    checkpointer=None,
):
    """
    Constructs and compiles the agentic RAG graph.

    Args:
        checkpointer: Pass MemorySaver() for multi-turn memory, or None
                       for stateless mode (used in tests).
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GROQ_API_KEY environment variable. "
            "Get a free key at https://console.groq.com/keys"
        )

    llm_client = Groq(api_key=api_key)
    storage = Storage(db_path)

    if qdrant_client is None:
        qdrant_client = QdrantClient(path=str(Path("data") / "qdrant_store"))
    if embed_model is None:
        embed_model = HFInferenceEmbedder(EMBEDDING_MODEL_NAME)
    if bm25 is None:
        bm25 = BM25Retriever.from_db(db_path)
    # reranker can be None; if None, reciprocal rank fusion (RRF) ordering is used directly
    pass

    # --- Bind resources to node functions via closures ---
    def analyzer(state):
        return nodes.analyze_query(state, llm_client)

    def retrieval(state):
        return nodes.execute_retrieval(
            state, qdrant_client, embed_model, bm25, reranker, llm_client, storage,
        )

    def generate(state):
        return nodes.generate_answer(state, llm_client)

    def verify(state):
        return nodes.verify_citations(state, llm_client)

    def oos(state):
        return nodes.out_of_scope(state)

    def save(state):
        return nodes.save_turn(state)

    # --- Wire the graph ---
    graph = StateGraph(AgentState)

    graph.add_node("analyze_query", analyzer)
    graph.add_node("execute_retrieval", retrieval)
    graph.add_node("generate", generate)
    graph.add_node("verify_citations", verify)
    graph.add_node("out_of_scope", oos)
    graph.add_node("save_turn", save)

    graph.set_entry_point("analyze_query")

    graph.add_conditional_edges("analyze_query", _route_after_analyzer, {
        "out_of_scope": "out_of_scope",
        "execute_retrieval": "execute_retrieval",
    })

    graph.add_edge("execute_retrieval", "generate")
    graph.add_edge("generate", "verify_citations")

    graph.add_conditional_edges("verify_citations", _route_after_verifier, {
        "execute_retrieval": "execute_retrieval",
        "save_turn": "save_turn",
    })

    graph.add_edge("out_of_scope", "save_turn")
    graph.add_edge("save_turn", END)

    return graph.compile(checkpointer=checkpointer)
