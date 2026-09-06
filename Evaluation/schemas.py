"""
Data schemas for the evaluation harness.

GoldenSample  — one hand-curated Q&A pair used as ground truth.
PipelineResult — one pipeline's output for a single question (answer + contexts + timing).
EvalReport    — aggregated RAGAS scores and latency stats for one pipeline run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class QueryType(str, Enum):
    """The four categories of questions spanning the evaluation set."""
    QUALITATIVE = "qualitative"       # prose from 10-K sections (risk factors, MD&A)
    QUANTITATIVE = "quantitative"     # exact numbers from XBRL facts
    COMPARATIVE = "comparative"       # multi-hop across companies or years
    OUT_OF_SCOPE = "out_of_scope"     # unrelated to filings — should be refused


@dataclass
class GoldenSample:
    """
    One labeled Q&A pair for the golden evaluation set.

    Fields:
        question:         The evaluation question as asked by the user.
        query_type:       One of QueryType enum values.
        reference_answer: The human-curated correct answer (ground truth).
        notes:            Optional: source chunk_id, XBRL fact_id, or curation notes.
    """
    question: str
    query_type: QueryType
    reference_answer: str
    notes: str = ""

    @classmethod
    def from_dict(cls, row: dict) -> "GoldenSample":
        return cls(
            question=row["question"],
            query_type=QueryType(row["query_type"]),
            reference_answer=row["reference_answer"],
            notes=row.get("notes", ""),
        )


@dataclass
class PipelineResult:
    """
    The output of running one pipeline on one question.

    Fields:
        question:      The original question.
        query_type:    Category from the golden sample.
        answer:        The pipeline's generated answer string.
        contexts:      List of retrieved context strings passed to the LLM.
        latency_ms:    Wall-clock time for the full pipeline call.
        retries:       Number of self-correction retries (agentic pipeline only).
        error:         If non-empty, the pipeline raised an exception.
    """
    question: str
    query_type: str
    answer: str
    contexts: list[str]
    latency_ms: float
    retries: int = 0
    error: str = ""


@dataclass
class EvalReport:
    """
    Aggregated RAGAS metrics and latency statistics for one pipeline run.
    """
    pipeline_name: str
    n_samples: int

    # RAGAS metrics (0.0 → 1.0, higher is better)
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # Latency (milliseconds)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0

    # Error rate
    error_rate: float = 0.0

    def as_dict(self) -> dict:
        return {
            "pipeline": self.pipeline_name,
            "n_samples": self.n_samples,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "error_rate": round(self.error_rate, 4),
        }
