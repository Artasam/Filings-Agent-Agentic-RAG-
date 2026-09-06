"""
Evaluation harness — runs both pipelines against the golden set.

Two runners are exposed:
  run_naive_pipeline   — uses RAG/pipeline_naive.py (baseline)
  run_agentic_pipeline — uses Agent/graph.py (full agentic system)

Each returns a list[PipelineResult] with the answer, retrieved contexts,
and wall-clock latency for every sample in the golden set.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from .schemas import GoldenSample, PipelineResult

logger = logging.getLogger("filingsagent.eval")


def run_naive_pipeline(
    samples: list[GoldenSample],
    qdrant_client,
    embed_model,
) -> list[PipelineResult]:
    """
    Runs every sample through the naive RAG baseline (dense retrieval + Gemini).
    No routing, no reranking, no self-correction.
    """
    from RAG.pipeline_naive import query as naive_query

    results: list[PipelineResult] = []
    for i, sample in enumerate(samples, 1):
        logger.info("Naive [%d/%d]: %s", i, len(samples), sample.question[:60])
        t0 = time.perf_counter()
        try:
            rag_result = naive_query(
                question=sample.question,
                qdrant_client=qdrant_client,
                embed_model=embed_model,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            contexts = [c.text for c in rag_result.retrieved_chunks]
            results.append(PipelineResult(
                question=sample.question,
                query_type=sample.query_type,
                answer=rag_result.answer,
                contexts=contexts,
                latency_ms=elapsed_ms,
            ))
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("Naive pipeline error on sample %d: %s", i, exc)
            results.append(PipelineResult(
                question=sample.question,
                query_type=sample.query_type,
                answer="",
                contexts=[],
                latency_ms=elapsed_ms,
                error=str(exc),
            ))
        time.sleep(2)  # Pacing between samples to respect API rate limits
    return results


def run_agentic_pipeline(
    samples: list[GoldenSample],
    agent,
) -> list[PipelineResult]:
    """
    Runs every sample through the full agentic pipeline
    (router → hybrid retrieval → citations → verifier).

    Each sample runs in its own thread_id so conversation memory does not
    bleed between evaluation samples.
    """
    results: list[PipelineResult] = []
    for i, sample in enumerate(samples, 1):
        logger.info("Agentic [%d/%d]: %s", i, len(samples), sample.question[:60])
        thread_id = f"eval-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}

        t0 = time.perf_counter()
        try:
            final_state = agent.invoke(
                {"question": sample.question, "messages": [], "retries": 0},
                config=config,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Collect context strings from both retrieved docs and SQL facts
            contexts: list[str] = []
            for doc in final_state.get("documents") or []:
                contexts.append(doc.get("text", ""))
            for fact in final_state.get("sql_results") or []:
                contexts.append(
                    f"{fact.get('concept')}={fact.get('value')} "
                    f"{fact.get('unit','')} FY{fact.get('fiscal_year','')}"
                )

            results.append(PipelineResult(
                question=sample.question,
                query_type=sample.query_type,
                answer=final_state.get("generation", ""),
                contexts=contexts,
                latency_ms=elapsed_ms,
                retries=final_state.get("retries", 0),
            ))
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("Agentic pipeline error on sample %d: %s", i, exc)
            results.append(PipelineResult(
                question=sample.question,
                query_type=sample.query_type,
                answer="",
                contexts=[],
                latency_ms=elapsed_ms,
                error=str(exc),
            ))
        time.sleep(2)  # Pacing between samples to respect API rate limits
    return results
