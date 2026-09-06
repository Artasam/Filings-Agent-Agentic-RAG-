"""
Offline tests for Phase 6: Evaluation framework.

Tests cover schemas, golden set loading, report generation, and harness
logic — all without real API calls or model downloads.

Run with:  python -m pytest tests/test_evaluation.py -v
"""
import csv
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Evaluation.report import generate_report, save_report
from Evaluation.schemas import EvalReport, GoldenSample, PipelineResult, QueryType


# ---------------------------------------------------------------------------
# Tests: GoldenSample schema
# ---------------------------------------------------------------------------

def test_golden_sample_from_dict_qualitative():
    row = {
        "question": "What are Apple's risk factors?",
        "query_type": "qualitative",
        "reference_answer": "Apple faces supply chain risks...",
        "notes": "Source: AAPL 10-K 2024",
    }
    sample = GoldenSample.from_dict(row)
    assert sample.query_type == QueryType.QUALITATIVE
    assert sample.question == "What are Apple's risk factors?"
    assert sample.notes == "Source: AAPL 10-K 2024"


def test_golden_sample_from_dict_quantitative():
    row = {
        "question": "What was AAPL revenue in 2024?",
        "query_type": "quantitative",
        "reference_answer": "$391 billion",
        "notes": "",
    }
    sample = GoldenSample.from_dict(row)
    assert sample.query_type == QueryType.QUANTITATIVE


def test_golden_sample_from_dict_comparative():
    row = {
        "question": "Compare Apple and Microsoft revenue.",
        "query_type": "comparative",
        "reference_answer": "Apple $391B, Microsoft $245B",
        "notes": "",
    }
    sample = GoldenSample.from_dict(row)
    assert sample.query_type == QueryType.COMPARATIVE


def test_golden_sample_from_dict_out_of_scope():
    row = {
        "question": "What's the weather?",
        "query_type": "out_of_scope",
        "reference_answer": "This question is outside the scope.",
        "notes": "",
    }
    sample = GoldenSample.from_dict(row)
    assert sample.query_type == QueryType.OUT_OF_SCOPE


# ---------------------------------------------------------------------------
# Tests: PipelineResult schema
# ---------------------------------------------------------------------------

def test_pipeline_result_defaults():
    result = PipelineResult(
        question="test?",
        query_type="qualitative",
        answer="Some answer.",
        contexts=["context1"],
        latency_ms=250.5,
    )
    assert result.retries == 0
    assert result.error == ""


def test_pipeline_result_with_error():
    result = PipelineResult(
        question="test?",
        query_type="quantitative",
        answer="",
        contexts=[],
        latency_ms=100.0,
        error="TimeoutError",
    )
    assert result.error == "TimeoutError"


# ---------------------------------------------------------------------------
# Tests: EvalReport schema
# ---------------------------------------------------------------------------

def test_eval_report_as_dict():
    report = EvalReport(
        pipeline_name="naive",
        n_samples=50,
        faithfulness=0.65,
        answer_relevancy=0.70,
        context_precision=0.50,
        context_recall=0.55,
        latency_p50_ms=320.0,
        latency_p95_ms=980.0,
        error_rate=0.02,
    )
    d = report.as_dict()
    assert d["pipeline"] == "naive"
    assert d["faithfulness"] == 0.65
    assert d["n_samples"] == 50


# ---------------------------------------------------------------------------
# Tests: Golden set CSV loading
# ---------------------------------------------------------------------------

def test_golden_set_loads_from_csv():
    """Valid CSV rows load correctly; [FILL IN] rows are skipped."""
    rows = [
        {"question": "What are AAPL risk factors?", "query_type": "qualitative",
         "reference_answer": "Supply chain risks...", "notes": ""},
        {"question": "AAPL revenue 2024?", "query_type": "quantitative",
         "reference_answer": "[FILL IN after ingesting data]", "notes": ""},
        {"question": "Weather?", "query_type": "out_of_scope",
         "reference_answer": "Out of scope.", "notes": ""},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "golden.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "query_type", "reference_answer", "notes"])
            writer.writeheader()
            writer.writerows(rows)

        # Simulate the load logic from run_eval.py
        samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "[FILL IN" not in row.get("reference_answer", ""):
                    samples.append(GoldenSample.from_dict(row))

        assert len(samples) == 2  # [FILL IN] row skipped
        assert samples[0].query_type == QueryType.QUALITATIVE
        assert samples[1].query_type == QueryType.OUT_OF_SCOPE


# ---------------------------------------------------------------------------
# Tests: Report generation
# ---------------------------------------------------------------------------

def _make_report(name, faith, rel, prec, rec, p50, p95, err):
    return EvalReport(
        pipeline_name=name, n_samples=20,
        faithfulness=faith, answer_relevancy=rel,
        context_precision=prec, context_recall=rec,
        latency_p50_ms=p50, latency_p95_ms=p95,
        error_rate=err,
    )


def test_generate_report_contains_all_metrics():
    naive = _make_report("naive", 0.65, 0.70, 0.50, 0.55, 320.0, 980.0, 0.02)
    agentic = _make_report("agentic", 0.88, 0.85, 0.78, 0.80, 750.0, 2100.0, 0.00)
    md = generate_report(naive, agentic)

    assert "Faithfulness" in md
    assert "Answer Relevancy" in md
    assert "Context Precision" in md
    assert "Context Recall" in md
    assert "0.6500" in md   # naive faithfulness
    assert "0.8800" in md   # agentic faithfulness
    assert "↑" in md        # positive delta


def test_generate_report_delta_direction():
    """Delta arrows must reflect actual direction of change."""
    naive = _make_report("naive", 0.90, 0.90, 0.90, 0.90, 100.0, 200.0, 0.0)
    agentic = _make_report("agentic", 0.70, 0.70, 0.70, 0.70, 150.0, 300.0, 0.0)
    md = generate_report(naive, agentic)
    # Agentic is WORSE here, so deltas should be negative
    assert "↓" in md


def test_save_report_creates_files():
    naive_r = _make_report("naive", 0.65, 0.70, 0.50, 0.55, 320.0, 980.0, 0.02)
    agentic_r = _make_report("agentic", 0.88, 0.85, 0.78, 0.80, 750.0, 2100.0, 0.00)
    md = generate_report(naive_r, agentic_r)

    naive_results = [PipelineResult("Q1?", "qualitative", "Answer1", ["ctx1"], 300.0)]
    agentic_results = [PipelineResult("Q1?", "qualitative", "Answer2", ["ctx1"], 700.0, retries=1)]

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "results"
        save_report(md, naive_results, agentic_results, output_dir)

        assert (output_dir / "eval_report.md").exists()
        assert (output_dir / "results_naive.csv").exists()
        assert (output_dir / "results_agentic.csv").exists()

        content = (output_dir / "eval_report.md").read_text()
        assert "FilingsAgent" in content
