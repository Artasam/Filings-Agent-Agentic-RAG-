"""
Thin, well-behaved client for SEC EDGAR's public endpoints.

Every outbound request goes through the shared RateLimiter and the
retry_with_backoff wrapper, so callers never need to think about rate
limits or transient failures -- they just call get_json()/get_bytes()
and get a result or a clear exception.
"""
from __future__ import annotations

import requests

from .config import Config
from .guardrails import RateLimiter, retry_with_backoff


class EdgarClient:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.user_agent, "Accept-Encoding": "gzip, deflate"})
        self.rate_limiter = RateLimiter(config.max_requests_per_second)

    def _get(self, url: str, context: str) -> requests.Response:
        def _do_request() -> requests.Response:
            self.rate_limiter.acquire()
            resp = self.session.get(url, timeout=self.config.request_timeout_seconds)
            resp.raise_for_status()
            return resp

        return retry_with_backoff(
            _do_request,
            max_retries=self.config.max_retries,
            base_backoff=self.config.base_backoff_seconds,
            max_backoff=self.config.max_backoff_seconds,
            context=context,
        )

    def get_json(self, url: str, context: str = "get_json") -> dict:
        return self._get(url, context).json()

    def get_bytes(self, url: str, context: str = "get_bytes") -> bytes:
        return self._get(url, context).content

    # --- higher-level EDGAR-specific calls -------------------------------
    def resolve_cik(self, ticker: str) -> str:
        """Returns the zero-padded 10-digit CIK for a ticker symbol."""
        mapping = self.get_json(self.config.edgar_ticker_map_url, context="ticker_map")
        ticker_upper = ticker.upper()
        for entry in mapping.values():
            if entry["ticker"].upper() == ticker_upper:
                return f"{entry['cik_str']:010d}"
        raise ValueError(f"Ticker '{ticker}' not found in SEC company_tickers.json")

    def get_submissions(self, cik: str) -> dict:
        url = self.config.edgar_submissions_url.format(cik=int(cik))
        return self.get_json(url, context=f"submissions[{cik}]")

    def get_company_facts(self, cik: str) -> dict:
        url = self.config.edgar_company_facts_url.format(cik=int(cik))
        return self.get_json(url, context=f"company_facts[{cik}]")

    def get_filing_document(self, cik: str, accession_no_dashes: str, primary_doc: str) -> bytes:
        # e.g. https://www.sec.gov/Archives/edgar/data/{cik}/{accn-no-dashes}/{primary_doc}
        url = f"{self.config.edgar_archives_base}/{int(cik)}/{accession_no_dashes}/{primary_doc}"
        return self.get_bytes(url, context=f"filing_doc[{cik}/{accession_no_dashes}]")
