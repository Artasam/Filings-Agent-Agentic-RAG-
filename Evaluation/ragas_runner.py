"""
RAGAS metric computation for pipeline results.

Uses RAGAS 0.1.21 API with Gemini 3.6 Flash as the LLM judge
and Google's embedding model for answer_relevancy embedding.

Metrics computed:
  faithfulness       -- are generated claims grounded in retrieved context?
  answer_relevancy   -- does the answer address the question?
  context_precision  -- is the retrieved context relevant to the question?
  context_recall     -- does the context cover the reference answer?
"""
from __future__ import annotations

import logging
import os

import numpy as np
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from .schemas import EvalReport, GoldenSample, PipelineResult

logger = logging.getLogger("filingsagent.eval")


def _build_ragas_llm_and_embeddings():
    """
    Constructs RAGAS 0.1.x-compatible LLM (Groq) and embedding wrappers backed by Gemini.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_groq import ChatGroq
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    groq_api_key = os.environ.get("GROQ_API_KEY")
    google_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not groq_api_key:
        raise EnvironmentError("Set GROQ_API_KEY environment variable for Groq LLM in RAGAS evaluation.")
    if not google_api_key:
        raise EnvironmentError("Set GOOGLE_API_KEY or GEMINI_API_KEY for embeddings in RAGAS evaluation.")

    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    llm = LangchainLLMWrapper(
        ChatGroq(model=model, groq_api_key=groq_api_key)
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=google_api_key
        )
    )
    return llm, embeddings


def _compute_latency_stats(results: list[PipelineResult]) -> tuple[float, float]:
    """Returns (p50_ms, p95_ms) from a list of pipeline results."""
    latencies = [r.latency_ms for r in results if not r.error]
    if not latencies:
        return 0.0, 0.0
    return float(np.percentile(latencies, 50)), float(np.percentile(latencies, 95))


def evaluate_pipeline(
    pipeline_name: str,
    results: list[PipelineResult],
    golden_samples: list[GoldenSample],
) -> EvalReport:
    """
    Runs RAGAS 0.1.x metrics on a list of PipelineResult objects and returns
    an EvalReport with all scores and latency stats.
    """
    # ragas 0.1.x API: uses Dataset from datasets library
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    llm, embeddings = _build_ragas_llm_and_embeddings()

    # Only evaluate samples without errors and with non-empty contexts
    valid_pairs = [
        (r, g)
        for r, g in zip(results, golden_samples)
        if not r.error and r.contexts and r.answer
    ]

    n_total = len(results)
    n_errors = sum(1 for r in results if r.error)
    error_rate = n_errors / n_total if n_total else 0.0

    if not valid_pairs:
        logger.warning("No valid samples for RAGAS evaluation — all errored or empty.")
        p50, p95 = _compute_latency_stats(results)
        return EvalReport(
            pipeline_name=pipeline_name,
            n_samples=n_total,
            error_rate=error_rate,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
        )

    # ragas 0.1.x expects a HuggingFace Dataset with these columns:
    ragas_data = {
        "question":   [r.question for r, _ in valid_pairs],
        "answer":     [r.answer for r, _ in valid_pairs],
        "contexts":   [r.contexts for r, _ in valid_pairs],
        "ground_truth": [g.reference_answer for _, g in valid_pairs],
    }
    dataset = Dataset.from_dict(ragas_data)

    logger.info(
        "Running RAGAS on %d samples for '%s' pipeline...",
        len(valid_pairs), pipeline_name,
    )
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    scores = ragas_result.to_pandas()
    p50, p95 = _compute_latency_stats(results)

    return EvalReport(
        pipeline_name=pipeline_name,
        n_samples=n_total,
        faithfulness=float(scores["faithfulness"].mean()),
        answer_relevancy=float(scores["answer_relevancy"].mean()),
        context_precision=float(scores["context_precision"].mean()),
        context_recall=float(scores["context_recall"].mean()),
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        error_rate=error_rate,
    )
