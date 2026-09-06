"""
Report generator — produces a Markdown comparison table and a CSV
from two EvalReport objects (naive vs. agentic).

Usage:
    from Evaluation.report import generate_report, save_report
    md = generate_report(naive_report, agentic_report)
    save_report(md, results_naive, results_agentic, output_dir)
"""
from __future__ import annotations

import csv
import datetime
from pathlib import Path

from .schemas import EvalReport, PipelineResult


def _delta_str(naive_val: float, agentic_val: float) -> str:
    """Returns a formatted delta string like '+0.15 ↑' or '-0.05 ↓'."""
    delta = agentic_val - naive_val
    arrow = "↑" if delta >= 0 else "↓"
    return f"{delta:+.4f} {arrow}"


def generate_report(naive: EvalReport, agentic: EvalReport) -> str:
    """
    Generates a Markdown report comparing naive vs. agentic pipeline metrics.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# FilingsAgent — Evaluation Report",
        f"\n_Generated: {now}_\n",
        "## RAGAS Metric Comparison\n",
        "| Metric | Naive RAG (Baseline) | Agentic RAG (Full) | Delta |",
        "|---|---|---|---|",
        f"| **Faithfulness** | {naive.faithfulness:.4f} | {agentic.faithfulness:.4f} | {_delta_str(naive.faithfulness, agentic.faithfulness)} |",
        f"| **Answer Relevancy** | {naive.answer_relevancy:.4f} | {agentic.answer_relevancy:.4f} | {_delta_str(naive.answer_relevancy, agentic.answer_relevancy)} |",
        f"| **Context Precision** | {naive.context_precision:.4f} | {agentic.context_precision:.4f} | {_delta_str(naive.context_precision, agentic.context_precision)} |",
        f"| **Context Recall** | {naive.context_recall:.4f} | {agentic.context_recall:.4f} | {_delta_str(naive.context_recall, agentic.context_recall)} |",
        "",
        "## Latency & Reliability\n",
        "| Metric | Naive RAG | Agentic RAG |",
        "|---|---|---|",
        f"| **p50 Latency** | {naive.latency_p50_ms:.0f} ms | {agentic.latency_p50_ms:.0f} ms |",
        f"| **p95 Latency** | {naive.latency_p95_ms:.0f} ms | {agentic.latency_p95_ms:.0f} ms |",
        f"| **Error Rate** | {naive.error_rate:.2%} | {agentic.error_rate:.2%} |",
        f"| **Samples Evaluated** | {naive.n_samples} | {agentic.n_samples} |",
        "",
        "## Notes\n",
        "- Faithfulness and context precision improvements are expected to be largest",
        "  due to structured citations and self-correction in the agentic pipeline.",
        "- The agentic pipeline is intentionally slower (routing + reranking + retries).",
        "  This tradeoff is by design and is a legitimate engineering finding.",
        "- Golden set size caveat: with n≈120 samples, expect wide confidence intervals.",
        "  Report these numbers honestly — do not overstate statistical precision.",
    ]
    return "\n".join(lines)


def save_report(
    report_md: str,
    naive_results: list[PipelineResult],
    agentic_results: list[PipelineResult],
    output_dir: Path,
) -> None:
    """
    Saves the Markdown report and per-sample CSV results to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Markdown report
    md_path = output_dir / "eval_report.md"
    md_path.write_text(report_md, encoding="utf-8")

    # Save per-sample CSV for both pipelines
    for pipeline_name, results in [("naive", naive_results), ("agentic", agentic_results)]:
        csv_path = output_dir / f"results_{pipeline_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["question", "query_type", "answer", "latency_ms", "retries", "error"],
            )
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "question": r.question,
                    "query_type": r.query_type,
                    "answer": r.answer[:200],
                    "latency_ms": round(r.latency_ms, 1),
                    "retries": r.retries,
                    "error": r.error,
                })

    print(f"Report saved to {output_dir}/")
    print(f"  - {md_path.name}")
    print(f"  - results_naive.csv")
    print(f"  - results_agentic.csv")
