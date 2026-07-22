"""
Central configuration for the FilingsAgent ingestion pipeline.

Everything that is "policy" (rate limits, retry behavior, paths, chunk sizing)
lives here so it can be tuned without touching pipeline logic.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # --- SEC EDGAR access -------------------------------------------------
    # SEC requires a descriptive User-Agent identifying you + contact info on
    # every request, or it will block you. Replace with your real info.
    # https://www.sec.gov/os/accessing-edgar-data
    user_agent: str = "FilingsAgent research-project artasambinrashid@gmail.com"

    edgar_submissions_url: str = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
    edgar_company_facts_url: str = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    edgar_ticker_map_url: str = "https://www.sec.gov/files/company_tickers.json"
    edgar_archives_base: str = "https://www.sec.gov/Archives/edgar/data"

    # SEC's stated fair-access limit is 10 requests/second across all its
    # endpoints. We default well under that to be a polite, reliable citizen
    # and avoid tripping their (undocumented) abuse detection.
    max_requests_per_second: float = 5.0

    # --- Retry / backoff ----------------------------------------------------
    max_retries: int = 5
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    request_timeout_seconds: float = 30.0

    # --- Concurrency ---------------------------------------------------------
    # Bounded worker pool. Kept modest by default; the rate limiter is the
    # real ceiling, this just controls how many filings are "in flight"
    # (parsing/chunking is CPU-bound and can run in parallel with I/O waits).
    max_workers: int = 4

    # --- Storage -------------------------------------------------------------
    data_dir: Path = Path("data")
    raw_filings_dir: Path = field(init=False)
    db_path: Path = field(init=False)
    quarantine_dir: Path = field(init=False)

    # --- Chunking --------------------------------------------------------------
    target_chunk_tokens: int = 650
    chunk_overlap_tokens: int = 100
    min_chunk_tokens: int = 40  # drop/merge fragments smaller than this

    # --- Scope for a single ingestion run ---------------------------------
    forms_to_ingest: tuple = ("10-K",)
    filings_per_company: int = 5  # most recent N filings of the given form(s)

    def __post_init__(self):
        object.__setattr__(self, "raw_filings_dir", self.data_dir / "raw_filings")
        object.__setattr__(self, "db_path", self.data_dir / "filingsagent.db")
        object.__setattr__(self, "quarantine_dir", self.data_dir / "quarantine")


CONFIG = Config()

# Sections we try to isolate out of a 10-K. Keys are canonical names used
# throughout storage/chunking; values are regex fragments matched against
# item headers in the cleaned document text (case-insensitive).
TARGET_SECTIONS = {
    "item_1_business": r"item\s+1\.?\s+business",
    "item_1a_risk_factors": r"item\s+1a\.?\s+risk\s+factors",
    "item_7_mdna": r"item\s+7\.?\s+management.?s\s+discussion",
    "item_7a_market_risk": r"item\s+7a\.?\s+quantitative\s+and\s+qualitative",
    "item_8_financial_statements": r"item\s+8\.?\s+financial\s+statements",
}

# XBRL concepts we pull for the structured (quantitative) retrieval path.
# These are standard us-gaap taxonomy tags present in nearly every filer.
XBRL_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetIncomeLoss",
    "EarningsPerShareDiluted",
    "EarningsPerShareBasic",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "ResearchAndDevelopmentExpense",
    "OperatingIncomeLoss",
    "CashAndCashEquivalentsAtCarryingValue",
]
