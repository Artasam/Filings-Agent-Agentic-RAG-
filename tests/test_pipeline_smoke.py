"""
End-to-end smoke test for the full Phase 1 pipeline (ingest_company),
using a mocked EdgarClient so it runs with zero network access. This
proves the orchestration wiring -- download -> validate -> parse -> chunk
-> store, plus XBRL fetch -> store, plus idempotent re-runs -- actually
works together, not just each piece in isolation.

Run with: python tests/test_pipeline_smoke.py
"""
import dataclasses
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Ingestion.config import CONFIG
from Ingestion.guardrails import Quarantine
from Ingestion.pipeline import ingest_company
from Ingestion.storage import Storage

FAKE_10K_HTML = ("""
<html><body>
<p>Item 1. Business</p>
<p>""" + ("We build software products for enterprises. " * 150) + """</p>
<p>Item 1A. Risk Factors</p>
<p>""" + ("Our results may be affected by competitive pressure. " * 150) + """</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>""" + ("Revenue grew due to strong subscription demand. " * 150) + """</p>
<p>Item 8. Financial Statements and Supplementary Data</p>
<p>""" + ("Refer to the notes to the consolidated financial statements. " * 150) + """</p>
</body></html>
""").encode("utf-8")

FAKE_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "10-K"],
            "accessionNumber": ["0000320193-24-000001", "0000320193-24-000050", "0000320193-23-000001"],
            "primaryDocument": ["aapl-10k-2024.htm", "aapl-10q-2024.htm", "aapl-10k-2023.htm"],
            "filingDate": ["2024-11-01", "2024-08-01", "2023-11-01"],
            "reportDate": ["2024-09-30", "2024-06-30", "2023-09-30"],
            "fiscalYear": [2024, 2024, 2023],
            "fiscalPeriod": ["FY", "Q3", "FY"],
        }
    }
}

FAKE_COMPANY_FACTS = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"val": 391000000000, "start": "2023-10-01", "end": "2024-09-30",
                         "fy": 2024, "fp": "FY", "form": "10-K", "accn": "0000320193-24-000001"}
                    ]
                }
            }
        }
    }
}


def build_mock_client():
    client = MagicMock()
    client.resolve_cik.return_value = "0000320193"
    client.get_submissions.return_value = FAKE_SUBMISSIONS
    client.get_company_facts.return_value = FAKE_COMPANY_FACTS
    # Only 10-K filings are in forms_to_ingest by default, so only 2 of the
    # 3 fake filings should actually be downloaded.
    client.get_filing_document.return_value = FAKE_10K_HTML
    return client


def main():
    with tempfile.TemporaryDirectory() as tmp:
        config = dataclasses.replace(
            CONFIG,
            data_dir=Path(tmp),
            filings_per_company=5,
            max_workers=2,
        )
        config.__post_init__()  # recompute derived paths under the new data_dir

        client = build_mock_client()
        storage = Storage(config.db_path)
        quarantine = Quarantine(config.quarantine_dir)

        try:
            ingest_company(client, storage, quarantine, config, "AAPL")

            counts = storage.counts()
            print("Counts after first run:", counts)
            assert counts["companies"] == 1
            assert counts["filings"] == 2, "expected only the two 10-K filings, not the 10-Q"
            assert counts["chunks"] > 0
            assert counts["xbrl_facts"] == 1
            assert client.get_filing_document.call_count == 2

            # --- idempotency: re-run should not re-download or duplicate rows ---
            ingest_company(client, storage, quarantine, config, "AAPL")
            counts2 = storage.counts()
            print("Counts after second (idempotent) run:", counts2)
            assert counts2 == counts, "re-running should not change row counts"
            assert client.get_filing_document.call_count == 2, "already-ingested filings must not be re-downloaded"

            print("\nSMOKE TEST PASSED")
        finally:
            storage.close()


if __name__ == "__main__":
    main()
