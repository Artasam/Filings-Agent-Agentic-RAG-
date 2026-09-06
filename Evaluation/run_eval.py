"""
End-to-end evaluation runner.

Usage:
    python -m Evaluation.run_eval \
        --golden data/eval/golden_set_template.csv \
        --db     data/filingsagent.db \
        --output data/eval/results/

This script:
1. Loads the golden set from a CSV.
2. Filters out any rows that still have [FILL IN] placeholder answers.
3. Runs both naive and agentic pipelines over every sample.
4. Computes RAGAS metrics.
5. Generates a Markdown report and CSV files in --output.

Note: Requires GOOGLE_API_KEY in the environment.
      Requires a populated Qdrant index (run the indexer first).
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root before anything else
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("filingsagent.eval")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_golden_set(csv_path: Path) -> list:
    """Loads and validates the golden set, skipping placeholder rows."""
    from Evaluation.schemas import GoldenSample

    samples = []
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "[FILL IN" in row.get("reference_answer", ""):
                skipped += 1
                continue
            try:
                samples.append(GoldenSample.from_dict(row))
            except Exception as e:
                logger.warning("Skipping invalid row '%s': %s", row.get("question", "")[:40], e)

    logger.info("Loaded %d valid samples (%d placeholders skipped)", len(samples), skipped)
    return samples


def main():
    parser = argparse.ArgumentParser(description="FilingsAgent Evaluation Runner")
    parser.add_argument("--golden", type=Path, required=True, help="Path to golden set CSV")
    parser.add_argument("--db", type=Path, default=Path("data/filingsagent.db"), help="SQLite DB path")
    parser.add_argument("--qdrant", type=Path, default=Path("data/qdrant_store"), help="Qdrant store path")
    parser.add_argument("--output", type=Path, default=Path("data/eval/results"), help="Output directory")
    parser.add_argument("--skip-naive", action="store_true", help="Skip naive pipeline evaluation")
    parser.add_argument("--skip-agentic", action="store_true", help="Skip agentic pipeline evaluation")
    args = parser.parse_args()

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: Set GROQ_API_KEY environment variable. Get a free key at https://console.groq.com/keys")
        sys.exit(1)

    google_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not google_api_key:
        print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable for Gemini embeddings.")
        sys.exit(1)

    samples = _load_golden_set(args.golden)
    if not samples:
        print("ERROR: No valid samples found. Fill in the reference_answer fields in the CSV first.")
        sys.exit(1)

    # --- Load shared resources ---
    from langgraph.checkpoint.memory import MemorySaver
    from qdrant_client import QdrantClient
    from sentence_transformers import CrossEncoder

    from Agent.graph import build_agent
    from Evaluation.harness import run_agentic_pipeline, run_naive_pipeline
    from Evaluation.ragas_runner import evaluate_pipeline
    from Evaluation.report import generate_report, save_report
    from RAG.hf_api_embedder import HFInferenceEmbedder
    from RAG.indexer import EMBEDDING_MODEL_NAME
    from RAG.retrievers.reranker import RERANKER_MODEL_NAME
    from RAG.retrievers.sparse import BM25Retriever

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    embed_model = HFInferenceEmbedder(EMBEDDING_MODEL_NAME)
    qdrant_client = QdrantClient(path=str(args.qdrant))
    bm25 = BM25Retriever.from_db(args.db)
    reranker = None  # Uses high-precision RRF fusion directly, saving 1.5GB RAM on CPU

    naive_results, agentic_results = [], []
    naive_report, agentic_report = None, None

    # --- Run Naive Pipeline ---
    if not args.skip_naive:
        logger.info("=== Running Naive RAG Pipeline (%d samples) ===", len(samples))
        naive_results = run_naive_pipeline(samples, qdrant_client, embed_model)
        naive_report = evaluate_pipeline("naive", naive_results, samples)
        logger.info("Naive — Faithfulness: %.4f | AnswerRel: %.4f | CtxPrec: %.4f | CtxRecall: %.4f",
                    naive_report.faithfulness, naive_report.answer_relevancy,
                    naive_report.context_precision, naive_report.context_recall)

    # --- Run Agentic Pipeline ---
    if not args.skip_agentic:
        logger.info("=== Running Agentic Pipeline (%d samples) ===", len(samples))
        agent = build_agent(
            db_path=args.db,
            qdrant_client=qdrant_client,
            embed_model=embed_model,
            bm25=bm25,
            reranker=reranker,
            api_key=groq_api_key,
            checkpointer=MemorySaver(),
        )
        agentic_results = run_agentic_pipeline(samples, agent)
        agentic_report = evaluate_pipeline("agentic", agentic_results, samples)
        logger.info("Agentic — Faithfulness: %.4f | AnswerRel: %.4f | CtxPrec: %.4f | CtxRecall: %.4f",
                    agentic_report.faithfulness, agentic_report.answer_relevancy,
                    agentic_report.context_precision, agentic_report.context_recall)

    # --- Generate Report ---
    if naive_report and agentic_report:
        report_md = generate_report(naive_report, agentic_report)
        save_report(report_md, naive_results, agentic_results, args.output)
        print("\n" + report_md)
    elif naive_report:
        print("Naive-only report:")
        print(naive_report.as_dict())
    elif agentic_report:
        print("Agentic-only report:")
        print(agentic_report.as_dict())


if __name__ == "__main__":
    main()
