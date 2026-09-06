"""
Agent state definition for the LangGraph state machine.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state that flows through every node in the agent graph.

    Fields:
        question:        The user's current question.
        messages:        Conversation history (append-only via operator.add).
        rewritten_query: The question after coreference resolution.
        sub_queries:     Decomposed sub-questions with per-query routing.
        documents:       Chunks retrieved by hybrid vector search.
        sql_results:     Rows returned by the XBRL SQL lookup.
        generation:      The LLM-generated answer.
        citations:       Per-claim citations with source IDs.
        route:           Tracks routing/grading decisions for conditional edges.
        retries:         How many verifier-triggered retries have occurred.
    """
    question: str
    messages: Annotated[list[dict], operator.add]
    rewritten_query: str
    sub_queries: list[dict]
    documents: list[dict]
    sql_results: list[dict]
    generation: str
    citations: list[dict]
    route: str
    retries: int
