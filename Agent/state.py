"""
Agent state definition for the LangGraph state machine.
"""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state that flows through every node in the agent graph.

    Fields:
        question:    The user's original question.
        route:       Classification from the router (vector_search | sql_search | out_of_scope).
        documents:   Chunks retrieved by hybrid vector search.
        sql_results: Rows returned by the XBRL SQL lookup.
        generation:  The LLM-generated answer.
        retries:     How many grader-triggered retries have occurred.
    """
    question: str
    route: str
    documents: list[dict]
    sql_results: list[dict]
    generation: str
    retries: int
