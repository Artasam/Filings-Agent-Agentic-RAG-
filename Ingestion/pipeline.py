"""
Orchestrates the end-to-end Phase 1 ingestion:

  for each company ticker:
      resolve CIK
      discover recent filings (10-K/10-Q) via submissions API
      for each filing (concurrently, bounded by max_workers):
          skip if already ingested (idempotent, resumable)
          download primary document
          validate raw content -> quarantine on failure
          parse into sections -> quarantine if too few sections found
          chunk each section -> store chunks
      fetch XBRL company facts -> normalize -> store

Every stage is wrapped so that ONE bad filing never aborts the whole run --
it gets logged and quarantined, and the pipeline moves on. This is the
property that lets this same code scale from 10 filings to millions: the
loop body is a pure function of (ticker, filing) with no shared mutable
state except the thread-safe Storage/RateLimiter, so scaling out is a
matter of raising max_workers / sharding the ticker list across machines,
not rewriting logic.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

from .chunker import chunk_section
from .config import Config
from .edgar_client import EdgarClient
from .guardrails import ContentValidator, Quarantine, logger
from .section_parser import html_to_clean_text, split_into_sections
from .storage import Storage, make_id
from .xbrl_fetcher import extract_facts


def _process_one_filing(
    client: EdgarClient,
    storage: Storage,
    quarantine: Quarantine,
    config: Config,
    cik: str,
    filing_meta: dict,
) -> str:
    """Returns a status string for logging/summary purposes."""
    accession_no = filing_meta["accession_no"]

    if storage.filing_exists(accession_no):
        storage.log("download", accession_no, "skipped", "already ingested")
        return "skipped"

    try:
        raw = client.get_filing_document(cik, filing_meta["accession_no_dashes"], filing_meta["primary_doc"])
    except Exception as e:  # noqa: BLE001 - deliberately broad: never let one filing kill the batch
        storage.log("download", accession_no, "error", str(e))
        logger.error("Download failed for %s: %s", accession_no, e)
        return "download_error"

    validation = ContentValidator.validate_filing_html(raw, filing_meta.get("source_url", ""))
    if not validation.ok:
        quarantine.record(accession_no, f"download_validation_failed: {validation.reason}", raw)
        storage.log("validate", accession_no, "error", validation.reason)
        return "quarantined"

    raw_path = config.raw_filings_dir / f"{accession_no}.html"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)

    import hashlib
    content_hash = hashlib.sha256(raw).hexdigest()

    storage.upsert_filing(
        accession_no=accession_no,
        cik=cik,
        form=filing_meta["form"],
        filing_date=filing_meta.get("filing_date"),
        fiscal_year=filing_meta.get("fiscal_year"),
        fiscal_period=filing_meta.get("fiscal_period"),
        primary_doc=filing_meta.get("primary_doc"),
        source_url=filing_meta.get("source_url"),
        content_sha256=content_hash,
        raw_path=str(raw_path),
        status="downloaded",
    )

    try:
        clean_text = html_to_clean_text(raw)
        sections = split_into_sections(clean_text)
    except Exception as e:  # noqa: BLE001
        quarantine.record(accession_no, f"parse_failed: {e}", raw)
        storage.log("parse", accession_no, "error", str(e))
        return "parse_error"

    if len(sections) < 2:
        quarantine.record(accession_no, f"too_few_sections_found ({len(sections)})", raw)
        storage.log("parse", accession_no, "error", f"only {len(sections)} sections found")
        return "quarantined"

    total_chunks = 0
    for section in sections:
        storage.upsert_section(accession_no, section.name, len(section.text))
        chunks = chunk_section(
            section.name,
            section.text,
            target_tokens=config.target_chunk_tokens,
            overlap_tokens=config.chunk_overlap_tokens,
            min_tokens=config.min_chunk_tokens,
        )
        rows = []
        for c in chunks:
            valid = ContentValidator.validate_chunk_text(c.text)
            if not valid.ok:
                continue  # silently-dropped low-signal fragments are fine (e.g. stray table junk)
            chunk_id = make_id(accession_no, c.section_name, str(c.chunk_index), c.text[:64])
            rows.append((chunk_id, accession_no, cik, c.section_name, c.chunk_index, c.token_count, c.text))
        storage.upsert_chunks(rows)
        total_chunks += len(rows)

    storage.mark_filing_status(accession_no, "chunked")
    storage.log("chunk", accession_no, "ok", f"{total_chunks} chunks across {len(sections)} sections")
    return "ok"


def ingest_company(client: EdgarClient, storage: Storage, quarantine: Quarantine, config: Config, ticker: str) -> None:
    try:
        cik = client.resolve_cik(ticker)
    except Exception as e:  # noqa: BLE001
        logger.error("Could not resolve CIK for %s: %s", ticker, e)
        storage.log("resolve_cik", ticker, "error", str(e))
        return

    storage.upsert_company(cik=cik, ticker=ticker.upper())

    try:
        submissions = client.get_submissions(cik)
    except Exception as e:  # noqa: BLE001
        logger.error("Could not fetch submissions for %s (cik=%s): %s", ticker, cik, e)
        storage.log("submissions", ticker, "error", str(e))
        return

    filings = _select_recent_filings(submissions, config.forms_to_ingest, config.filings_per_company)
    logger.info("%s (cik=%s): %d filings selected for ingestion", ticker, cik, len(filings))

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {
            pool.submit(_process_one_filing, client, storage, quarantine, config, cik, f): f["accession_no"]
            for f in filings
        }
        for future in as_completed(futures):
            accession_no = futures[future]
            try:
                status = future.result()
                logger.info("Filing %s: %s", accession_no, status)
            except Exception as e:  # noqa: BLE001
                logger.error("Unhandled error processing %s: %s", accession_no, e)
                storage.log("process_filing", accession_no, "error", str(e))

    # --- XBRL structured facts (one call per company, not per filing) -----
    try:
        facts_json = client.get_company_facts(cik)
        rows = list(extract_facts(cik, facts_json))
        n = storage.upsert_xbrl_facts(rows)
        storage.log("xbrl", cik, "ok", f"{n} facts")
        logger.info("%s: %d XBRL facts stored", ticker, n)
    except Exception as e:  # noqa: BLE001
        logger.error("XBRL fetch failed for %s: %s", ticker, e)
        storage.log("xbrl", cik, "error", str(e))


def _select_recent_filings(submissions: dict, forms: Iterable[str], limit_per_company: int) -> list[dict]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms_list = recent.get("form", [])
    accession_list = recent.get("accessionNumber", [])
    primary_doc_list = recent.get("primaryDocument", [])
    filing_date_list = recent.get("filingDate", [])
    report_date_list = recent.get("reportDate", [])
    fy_list = recent.get("fiscalYear", [None] * len(forms_list))
    fp_list = recent.get("fiscalPeriod", [None] * len(forms_list))

    selected = []
    for i, form in enumerate(forms_list):
        if form not in forms:
            continue
        accession_no = accession_list[i]
        accession_no_dashes = accession_no.replace("-", "")
        selected.append({
            "accession_no": accession_no,
            "accession_no_dashes": accession_no_dashes,
            "form": form,
            "primary_doc": primary_doc_list[i],
            "filing_date": filing_date_list[i],
            "report_date": report_date_list[i] if i < len(report_date_list) else None,
            "fiscal_year": fy_list[i] if i < len(fy_list) else None,
            "fiscal_period": fp_list[i] if i < len(fp_list) else None,
        })
        if len(selected) >= limit_per_company:
            break
    return selected


def run(config: Config, tickers: list[str]) -> dict:
    client = EdgarClient(config)
    storage = Storage(config.db_path)
    quarantine = Quarantine(config.quarantine_dir)

    try:
        for ticker in tickers:
            logger.info("=== Ingesting %s ===", ticker)
            ingest_company(client, storage, quarantine, config, ticker)

        counts = storage.counts()
        logger.info("Run complete. Row counts: %s", counts)
        return counts
    finally:
        storage.close()
