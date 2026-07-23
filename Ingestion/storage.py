"""
Storage layer.

SQLite is the right choice for a portfolio-scale run (hundreds to tens of
thousands of filings / low millions of chunks) -- zero infra, single file,
trivial to inspect. Every write here is an idempotent UPSERT keyed by a
stable natural key (accession number, or a content hash for chunks), so
re-running the pipeline over the same filings never creates duplicates.

SCALING PAST THIS: see README "Scaling beyond SQLite" -- the schema below
is intentionally written so that migrating to Postgres is close to a
drop-in swap (same tables/keys), because the natural-key + hash-based
idempotency pattern is what matters, not the specific database.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    cik TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    name TEXT
);

CREATE TABLE IF NOT EXISTS filings (
    accession_no TEXT PRIMARY KEY,
    cik TEXT NOT NULL,
    form TEXT NOT NULL,
    filing_date TEXT,
    fiscal_year INTEGER,
    fiscal_period TEXT,
    primary_doc TEXT,
    source_url TEXT,
    content_sha256 TEXT,
    raw_path TEXT,
    status TEXT DEFAULT 'downloaded',   -- downloaded -> parsed -> chunked -> failed
    FOREIGN KEY (cik) REFERENCES companies(cik)
);

CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    accession_no TEXT NOT NULL,
    section_name TEXT NOT NULL,
    char_count INTEGER,
    UNIQUE(accession_no, section_name),
    FOREIGN KEY (accession_no) REFERENCES filings(accession_no)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,          -- sha256 of (accession_no, section, index, text)
    accession_no TEXT NOT NULL,
    cik TEXT NOT NULL,
    section_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER,
    text TEXT NOT NULL,
    FOREIGN KEY (accession_no) REFERENCES filings(accession_no)
);

CREATE TABLE IF NOT EXISTS xbrl_facts (
    fact_id TEXT PRIMARY KEY,           -- sha256 of (cik, concept, unit, start, end, val, accn)
    cik TEXT NOT NULL,
    concept TEXT NOT NULL,
    unit TEXT,
    value REAL,
    period_start TEXT,
    period_end TEXT,
    fiscal_year INTEGER,
    fiscal_period TEXT,
    form TEXT,
    accession_no TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT DEFAULT CURRENT_TIMESTAMP,
    stage TEXT,
    item_id TEXT,
    status TEXT,       -- ok | skipped | error
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_cik ON chunks(cik);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(accession_no, section_name);
CREATE INDEX IF NOT EXISTS idx_xbrl_cik_concept ON xbrl_facts(cik, concept);
CREATE INDEX IF NOT EXISTS idx_filings_cik ON filings(cik);
"""


def make_id(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


class Storage:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("PRAGMA journal_mode=WAL;")   # concurrent readers + 1 writer, safe at scale
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        with self.lock:
            self.conn.close()

    @contextmanager
    def cursor(self):
        with self.lock:
            cur = self.conn.cursor()
            try:
                yield cur
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
            finally:
                cur.close()

    # --- writes -------------------------------------------------------------
    def upsert_company(self, cik: str, ticker: str, name: str | None = None) -> None:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO companies (cik, ticker, name) VALUES (?, ?, ?) "
                "ON CONFLICT(cik) DO UPDATE SET ticker=excluded.ticker, name=excluded.name",
                (cik, ticker, name),
            )

    def filing_exists(self, accession_no: str) -> bool:
        with self.cursor() as cur:
            cur.execute("SELECT 1 FROM filings WHERE accession_no = ?", (accession_no,))
            return cur.fetchone() is not None

    def upsert_filing(self, **fields) -> None:
        cols = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        updates = ", ".join(f"{k}=excluded.{k}" for k in fields if k != "accession_no")
        with self.cursor() as cur:
            cur.execute(
                f"INSERT INTO filings ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(accession_no) DO UPDATE SET {updates}",
                tuple(fields.values()),
            )

    def mark_filing_status(self, accession_no: str, status: str) -> None:
        """Updates just the status of an already-inserted filing row."""
        with self.cursor() as cur:
            cur.execute("UPDATE filings SET status = ? WHERE accession_no = ?", (status, accession_no))

    def upsert_section(self, accession_no: str, section_name: str, char_count: int) -> None:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO sections (accession_no, section_name, char_count) VALUES (?, ?, ?) "
                "ON CONFLICT(accession_no, section_name) DO UPDATE SET char_count=excluded.char_count",
                (accession_no, section_name, char_count),
            )

    def upsert_chunks(self, chunk_rows: Iterable[tuple]) -> int:
        """chunk_rows: iterable of (chunk_id, accession_no, cik, section_name, chunk_index, token_count, text)"""
        rows = list(chunk_rows)
        with self.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (chunk_id, accession_no, cik, section_name, chunk_index, token_count, text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(chunk_id) DO NOTHING",
                rows,
            )
        return len(rows)

    def upsert_xbrl_facts(self, fact_rows: Iterable[tuple]) -> int:
        """fact_rows: iterable of (fact_id, cik, concept, unit, value, period_start, period_end,
        fiscal_year, fiscal_period, form, accession_no)"""
        rows = list(fact_rows)
        with self.cursor() as cur:
            cur.executemany(
                "INSERT INTO xbrl_facts (fact_id, cik, concept, unit, value, period_start, period_end, "
                "fiscal_year, fiscal_period, form, accession_no) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(fact_id) DO NOTHING",
                rows,
            )
        return len(rows)

    def log(self, stage: str, item_id: str, status: str, detail: str = "") -> None:
        with self.cursor() as cur:
            cur.execute(
                "INSERT INTO ingestion_log (stage, item_id, status, detail) VALUES (?, ?, ?, ?)",
                (stage, item_id, status, detail),
            )

    # --- reads (used by tests / sanity checks / later retrieval code) ------
    def counts(self) -> dict:
        with self.cursor() as cur:
            out = {}
            for table in ("companies", "filings", "sections", "chunks", "xbrl_facts"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                out[table] = cur.fetchone()[0]
            return out

    def get_cik_for_ticker(self, ticker: str) -> str | None:
        """Resolves a ticker symbol to its CIK."""
        with self.cursor() as cur:
            cur.execute("SELECT cik FROM companies WHERE ticker = ?", (ticker.upper(),))
            row = cur.fetchone()
            return row[0] if row else None

    def get_xbrl_facts(
        self, cik: str, concept: str | None = None, fiscal_year: int | None = None,
    ) -> list[dict]:
        """Queries structured XBRL facts by CIK, with optional concept/year filters."""
        with self.cursor() as cur:
            sql = "SELECT concept, unit, value, fiscal_year, fiscal_period, form FROM xbrl_facts WHERE cik = ?"
            params: list = [cik]
            if concept:
                sql += " AND concept = ?"
                params.append(concept)
            if fiscal_year:
                sql += " AND fiscal_year = ?"
                params.append(fiscal_year)
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

