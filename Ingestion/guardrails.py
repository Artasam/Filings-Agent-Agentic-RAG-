"""
Guardrails for the ingestion pipeline.

These are the boring-but-critical pieces that make an ingestion pipeline
survive contact with a real external API and scale past a handful of
documents without silently corrupting data or getting your IP banned:

- RateLimiter: token-bucket limiter so we never exceed SEC's fair-access
  policy, regardless of how many workers are running concurrently.
- retry_with_backoff: exponential backoff + jitter, retries only on
  transient errors (timeouts, 429, 5xx) and never on 4xx client errors
  that won't fix themselves.
- ContentValidator: rejects empty, truncated, non-HTML, or suspiciously
  small documents before they ever reach parsing/chunking.
- Quarantine: anything that fails validation or parsing is written to a
  quarantine directory with the reason, instead of being silently dropped
  or crashing the whole run. This is what lets a pipeline scale to
  millions of documents unattended -- a handful of malformed filings
  should never take down the batch.
- StructuredLogger: consistent, greppable log lines for observability.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import requests

T = TypeVar("T")

logger = logging.getLogger("filingsagent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.

    Shared across all workers so that "max_requests_per_second" is a true
    ceiling on total outbound request rate, not a per-worker limit that
    multiplies with concurrency.
    """

    def __init__(self, requests_per_second: float):
        self._interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed_time = time.monotonic()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed_time - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed_time = max(now, self._next_allowed_time) + self._interval


# --------------------------------------------------------------------------
# Retry / backoff
# --------------------------------------------------------------------------
class NonRetryableError(Exception):
    """Raised for client errors (4xx other than 429) that retrying won't fix."""


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been used up."""


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_backoff: float,
    max_backoff: float,
    retryable_status_codes: frozenset = frozenset({429, 500, 502, 503, 504}),
    context: str = "",
) -> T:
    """
    Calls fn() with exponential backoff + full jitter on transient failures.

    fn should raise requests.HTTPError / requests.RequestException on
    failure (this is what requests.Response.raise_for_status() does).
    """
    attempt = 0
    while True:
        try:
            return fn()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is not None and status not in retryable_status_codes:
                raise NonRetryableError(f"{context}: HTTP {status} (not retryable)") from e
            attempt += 1
            if attempt > max_retries:
                raise RetryExhaustedError(
                    f"{context}: exhausted {max_retries} retries (last status={status})"
                ) from e
            sleep_for = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
            sleep_for = random.uniform(0, sleep_for)  # full jitter
            logger.warning(
                "%s: transient error (status=%s), retry %d/%d in %.1fs",
                context, status, attempt, max_retries, sleep_for,
            )
            time.sleep(sleep_for)
        except requests.RequestException as e:
            attempt += 1
            if attempt > max_retries:
                raise RetryExhaustedError(
                    f"{context}: exhausted {max_retries} retries (last error={e})"
                ) from e
            sleep_for = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
            sleep_for = random.uniform(0, sleep_for)
            logger.warning(
                "%s: network error (%s), retry %d/%d in %.1fs",
                context, e, attempt, max_retries, sleep_for,
            )
            time.sleep(sleep_for)


# --------------------------------------------------------------------------
# Content validation + quarantine
# --------------------------------------------------------------------------
@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""


class ContentValidator:
    """
    Cheap, fast sanity checks run BEFORE a document is parsed/chunked.
    Catches the common failure modes seen at scale: empty responses,
    HTML error pages served with a 200 status, truncated downloads,
    and non-HTML content served on an HTML URL.
    """

    MIN_BYTES = 2_000  # a real 10-K is hundreds of KB+; anything tiny is suspect

    @classmethod
    def validate_filing_html(cls, content: bytes, url: str) -> ValidationResult:
        if not content:
            return ValidationResult(False, "empty response body")
        if len(content) < cls.MIN_BYTES:
            return ValidationResult(False, f"suspiciously small ({len(content)} bytes)")
        lowered = content[:2000].lower()
        if b"<html" not in lowered and b"<!doctype html" not in lowered and b"<sec-document" not in lowered:
            return ValidationResult(False, "does not look like HTML/SGML filing content")
        if b"you are unable to access" in lowered or b"error 403" in lowered.lower():
            return ValidationResult(False, "looks like a block/error page, not filing content")
        return ValidationResult(True)

    @classmethod
    def validate_chunk_text(cls, text: str, min_chars: int = 30) -> ValidationResult:
        stripped = text.strip()
        if len(stripped) < min_chars:
            return ValidationResult(False, "chunk too short after cleaning")
        alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
        if alpha_ratio < 0.3:
            return ValidationResult(False, "chunk is mostly non-alphabetic (likely markup/table noise)")
        return ValidationResult(True)


class Quarantine:
    """
    Anything that fails validation/parsing gets written here with the
    reason, keyed by a stable id, instead of crashing the batch or being
    silently dropped. At scale, some fraction of documents WILL be
    malformed -- the pipeline's job is to keep going and leave a trail.
    """

    def __init__(self, quarantine_dir: Path):
        self.dir = quarantine_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def record(self, item_id: str, reason: str, payload: bytes | str = b"") -> None:
        logger.error("QUARANTINED %s: %s", item_id, reason)
        meta_path = self.dir / f"{item_id}.reason.txt"
        meta_path.write_text(reason, encoding="utf-8")
        if payload:
            raw_path = self.dir / f"{item_id}.raw"
            mode = "wb" if isinstance(payload, bytes) else "w"
            with open(raw_path, mode) as f:
                f.write(payload)
