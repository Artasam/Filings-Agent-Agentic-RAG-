"""
Offline tests -- no network access required. These exercise every piece of
Phase 1 logic that doesn't require hitting SEC EDGAR: HTML section parsing,
chunking, SQLite idempotency, and guardrail validators. Run with:

    python -m pytest tests/test_offline.py -v
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Ingestion.chunker import chunk_section, count_tokens
from Ingestion.config import Config
from Ingestion.guardrails import ContentValidator, RateLimiter, retry_with_backoff, NonRetryableError, RetryExhaustedError
from Ingestion.section_parser import html_to_clean_text, split_into_sections
from Ingestion.storage import Storage, make_id
from Ingestion.xbrl_fetcher import extract_facts
import requests


SAMPLE_10K_HTML = """
<html><body>
<p>TABLE OF CONTENTS</p>
<p>Item 1A. Risk Factors 14</p>
<p>Item 7. Management's Discussion and Analysis 40</p>
<hr/>
<p>Item 1. Business</p>
<p>We design, manufacture and sell widgets globally. """ + ("Our widgets are great. " * 80) + """</p>
<p>Item 1A. Risk Factors</p>
<p>Our business is subject to numerous risks. """ + ("Competition could harm our margins. " * 120) + """</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Revenue increased year over year due to strong demand. """ + ("Costs also rose modestly. " * 100) + """</p>
<p>Item 8. Financial Statements and Supplementary Data</p>
<p>See consolidated financial statements below. """ + ("Balance sheet detail follows. " * 60) + """</p>
</body></html>
"""


def test_html_to_clean_text_strips_tags():
    text = html_to_clean_text(SAMPLE_10K_HTML.encode("utf-8"))
    assert "<p>" not in text
    assert "widgets" in text.lower()


def test_split_into_sections_finds_real_sections_not_toc():
    text = html_to_clean_text(SAMPLE_10K_HTML.encode("utf-8"))
    sections = split_into_sections(text)
    names = {s.name for s in sections}
    assert "item_1_business" in names
    assert "item_1a_risk_factors" in names
    assert "item_7_mdna" in names
    assert "item_8_financial_statements" in names
    # The real Risk Factors section body should be much longer than the
    # 3-word ToC line ("Item 1A. Risk Factors 14"), proving we picked the
    # section body and not the table-of-contents entry.
    risk_section = next(s for s in sections if s.name == "item_1a_risk_factors")
    assert len(risk_section.text) > 500


def test_chunk_section_respects_target_and_overlap():
    text = "Sentence number %d provides some filler content for chunking. " 
    long_text = "\n".join(text % i for i in range(200))
    chunks = chunk_section("item_1a_risk_factors", long_text, target_tokens=100, overlap_tokens=20, min_tokens=10)
    assert len(chunks) > 1
    for c in chunks[:-1]:
        # allow some slack since we pack whole paragraphs/sentences
        assert c.token_count <= 130
    # overlap: end of chunk N should share words with start of chunk N+1
    if len(chunks) >= 2:
        tail_words = set(chunks[0].text.split()[-15:])
        head_words = set(chunks[1].text.split()[:30])
        assert tail_words & head_words, "expected some overlapping tokens between consecutive chunks"


def test_chunk_section_handles_oversized_single_paragraph():
    huge_paragraph = "This is one sentence. " * 500  # no newlines, one giant paragraph
    chunks = chunk_section("item_7_mdna", huge_paragraph, target_tokens=80, overlap_tokens=10, min_tokens=10)
    assert len(chunks) > 3
    assert all(c.token_count <= 100 for c in chunks)


def test_content_validator_rejects_tiny_and_error_pages():
    assert not ContentValidator.validate_filing_html(b"", "url").ok
    assert not ContentValidator.validate_filing_html(b"short", "url").ok
    fake_block_page = b"<html>You are unable to access sec.gov</html>" + b" " * 3000
    assert not ContentValidator.validate_filing_html(fake_block_page, "url").ok
    real_looking = b"<html><body>" + b"Item 1. Business. " * 500 + b"</body></html>"
    assert ContentValidator.validate_filing_html(real_looking, "url").ok


def test_content_validator_rejects_markup_noise_chunks():
    assert not ContentValidator.validate_chunk_text("| | | 12 34 || -- | |").ok
    assert ContentValidator.validate_chunk_text("This is a normal sentence about revenue growth.").ok


def test_rate_limiter_enforces_interval():
    limiter = RateLimiter(requests_per_second=5)  # 200ms between calls
    start = time.monotonic()
    for _ in range(3):
        limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.4 - 0.05  # 2 intervals of 0.2s, small tolerance


def test_retry_with_backoff_raises_nonretryable_on_404():
    class FakeResponse:
        status_code = 404
    def flaky():
        err = requests.HTTPError()
        err.response = FakeResponse()
        raise err
    try:
        retry_with_backoff(flaky, max_retries=3, base_backoff=0.01, max_backoff=0.02, context="test")
        assert False, "expected NonRetryableError"
    except NonRetryableError:
        pass


def test_retry_with_backoff_succeeds_after_transient_failures():
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            err = requests.HTTPError()
            class R: status_code = 503
            err.response = R()
            raise err
        return "ok"
    result = retry_with_backoff(flaky, max_retries=5, base_backoff=0.01, max_backoff=0.02, context="test")
    assert result == "ok"
    assert calls["n"] == 3


def test_storage_upserts_are_idempotent(tmp_path):
    storage = Storage(tmp_path / "test.db")
    storage.upsert_company(cik="0000000001", ticker="TEST")
    storage.upsert_filing(accession_no="0000000001-24-000001", cik="0000000001", form="10-K",
                           filing_date="2024-01-01", status="downloaded")
    assert storage.filing_exists("0000000001-24-000001")

    chunk_id = make_id("acc1", "sec1", "0", "hello world")
    rows = [(chunk_id, "0000000001-24-000001", "0000000001", "item_1_business", 0, 2, "hello world")]
    n1 = storage.upsert_chunks(rows)
    n2 = storage.upsert_chunks(rows)  # re-insert same rows: should not duplicate
    assert n1 == 1 and n2 == 1
    counts = storage.counts()
    assert counts["chunks"] == 1
    assert counts["filings"] == 1


def test_extract_facts_flattens_xbrl_payload():
    fake_payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"val": 1000, "start": "2022-01-01", "end": "2022-12-31",
                             "fy": 2022, "fp": "FY", "form": "10-K", "accn": "0001-22-000001"},
                            {"val": None, "start": "2023-01-01", "end": "2023-12-31",
                             "fy": 2023, "fp": "FY", "form": "10-K", "accn": "0001-23-000001"},
                        ]
                    }
                }
            }
        }
    }
    rows = list(extract_facts("0000000001", fake_payload))
    assert len(rows) == 1  # the None value row must be skipped
    assert rows[0][2] == "Revenues"
    assert rows[0][4] == 1000.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
