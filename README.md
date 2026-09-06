# FilingsAgent — Phases 1 to 4: From Ingestion to Agentic RAG

This repository covers the complete end-to-end implementation of the FilingsAgent blueprint.

- **Phase 1 (Ingestion)**: Download → validate → section-parse → chunk → store for SEC 10-K/10-Q filings, plus a structured XBRL facts fetcher.
- **Phase 2 (Naive RAG)**: Baseline retrieval generation via Qdrant (`nomic-embed-text-v1.5`) and Gemini 3.6 Flash.
- **Phase 3 (Hybrid RAG)**: Improved retrieval accuracy by combining Dense Vector Search (Qdrant) with Sparse Keyword Search (BM25) using Reciprocal Rank Fusion (RRF), followed by a Cross-Encoder reranker (`BAAI/bge-reranker-base`).
- **Phase 4 (Agentic Routing)**: LangGraph state machine that intelligently routes queries. It sends qualitative questions to the Hybrid Vector Search and quantitative financial queries to a direct SQL lookup (XBRL database), complete with an LLM grader and self-correction retry loop.

## Install

```bash
pip install -r requirements.txt
```

## Run the offline tests (no network required)

```bash
# Test Phase 1 (Ingestion)
python -m pytest tests/test_offline.py -v
python -m pytest tests/test_pipeline_smoke.py -v

# Test Phase 2, 3, & 4 (RAG Baseline, Hybrid, Agentic)
python -m pytest tests/test_rag_baseline.py tests/test_hybrid_rag.py tests/test_agent.py -v
```

`test_offline.py` unit-tests parsing/chunking/storage/guardrails in
isolation. `test_pipeline_smoke.py` runs the **entire orchestration**
(`ingest_company`) against a mocked EDGAR client — no network — and
asserts correct filtering (only requested forms), correct chunk/fact
counts, and, critically, **idempotency** (running it twice doesn't
re-download or duplicate a single row). This test is what caught a real
bug during development (a partial-column UPSERT tripping a NOT NULL
constraint) — keep it in your CI.

## Run it for real

SEC requires a descriptive `User-Agent` on every request — set yours:

```bash
python -m Ingestion.cli ingest \
    --tickers AAPL MSFT GOOGL \
    --forms 10-K \
    --per-company 3 \
    --user-agent "Your Name your_real_email@domain.com"
```

This will populate `data/filingsagent.db` (SQLite) and `data/raw_filings/`.
Re-running the same command is safe and cheap — already-ingested filings
are skipped.

### RAG Generation (Gemini)

For the LLM generation step, you need a Gemini API key. Set it in your environment or `.env` file:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

Then seed your eval golden set (Section 4 of the blueprint):

```bash
python -m Ingestion.cli seed-eval --n 40 --out data/eval_seed.csv
```

Open the CSV and hand-write `question` / `expected_answer` for each row.

## What's actually in here

```
Agent/
  graph.py           LangGraph state machine wiring (router -> tools -> generate -> grade)
  nodes.py           pure Python functions for each node in the graph
  state.py           TypedDict defining the state passed through the graph
Ingestion/
  config.py          all tunables: rate limits, retry policy, chunk sizing, target sections/concepts
  guardrails.py      RateLimiter, retry_with_backoff, ContentValidator, Quarantine, logging
  edgar_client.py    rate-limited + retrying wrapper around SEC's public endpoints
  section_parser.py  splits raw 10-K HTML into Item 1 / 1A / 7 / 7A / 8
  chunker.py         paragraph-aware, token-budgeted chunking with overlap
  xbrl_fetcher.py    flattens SEC's nested XBRL company-facts JSON into rows
  storage.py         SQLite schema + idempotent upserts (WAL mode) + SQL retrieval
  pipeline.py        orchestrates the above with bounded concurrency
  eval_seed.py       samples stored chunks/facts into an eval-authoring template
  cli.py             `ingest` and `seed-eval` commands
RAG/
  indexer.py         embeds chunks with nomic-embed-text-v1.5 and upserts to Qdrant
  generation.py      LLM generation and context formatting
  pipeline_naive.py  baseline dense retrieval + Gemini generation
  pipeline_hybrid.py dense + sparse + RRF + reranking + generation
  schema.py          core data types (RetrievedChunk, RAGResult)
  retrievers/
    dense.py         Qdrant vector search
    sparse.py        BM25 exact keyword matching
    fusion.py        Reciprocal Rank Fusion (RRF)
    reranker.py      Cross-encoder scoring (BAAI/bge-reranker-base)
tests/
  test_offline.py         unit tests for Phase 1, no network
  test_pipeline_smoke.py  full pipeline run against a mocked EDGAR client
  test_rag_baseline.py    offline tests for Qdrant indexing and naive logic
  test_hybrid_rag.py      offline tests for BM25, RRF, and cross-encoder reranking
  test_agent.py           offline tests for LangGraph nodes, routing, and self-correction
```

## Guardrails baked in (not bolted on)

- **Rate limiting**: a single shared token-bucket `RateLimiter` caps total
  outbound request rate regardless of worker count, well under SEC's
  documented fair-access ceiling.
- **Retry policy**: exponential backoff + full jitter on timeouts/429/5xx;
  4xx client errors fail fast (`NonRetryableError`) instead of retrying
  something that will never succeed.
- **Content validation before parsing**: empty bodies, truncated
  downloads, and block/error pages served with HTTP 200 are caught before
  they ever reach the parser.
- **Quarantine, not crash**: any filing that fails validation or yields
  fewer than 2 recognizable sections is written to `data/quarantine/`
  with the failure reason, and the batch keeps going. At scale, a
  fraction of documents will always be malformed — the pipeline's job is
  to isolate and log that, not stop.
- **Thread-safe writes**: `pipeline.py` shares one `Storage` connection
  across a `ThreadPoolExecutor`. `check_same_thread=False` only disables
  Python's same-thread *check* — it doesn't add real synchronization — so
  `Storage` serializes all cursor/commit access through a `threading.Lock`.
  Without it, concurrent writers can corrupt the driver's internal state;
  this doesn't necessarily reproduce on every OS or every run, which makes
  it easy to miss until it isn't. Call `storage.close()` (or use `Storage`
  as a context manager) when a run finishes — on Windows an open
  connection holds a file lock, so anything that tries to move/delete the
  `.db` file afterwards will fail until it's released.
- **Idempotency**: every write is a natural-key or content-hash keyed
  UPSERT (`filings.accession_no`, `chunks.chunk_id` = sha256 of content,
  `xbrl_facts.fact_id` = sha256 of the fact tuple). Re-running the
  pipeline over the same data is always safe.
- **Observability**: every stage (download/validate/parse/chunk/xbrl)
  writes to an `ingestion_log` table, so you can query success/error
  rates per stage without re-running anything.

## Scaling beyond this (millions of documents)

This code is structured so scaling is a matter of turning dials, not
rewriting logic:

1. **Concurrency**: `max_workers` bounds in-flight filings; the real
   ceiling is the shared `RateLimiter`, so you can safely raise workers
   without risking a burst that gets you rate-limited.
2. **Sharding**: `ingest_company` is a pure function of `(ticker, config)`
   with no shared mutable state except the thread-safe `Storage` and
   `RateLimiter`. To scale out across machines, shard the ticker list and
   point each shard at its own SQLite file (or a shared Postgres — see
   next point), then merge.
3. **Storage**: SQLite is right for a portfolio-scale run (thousands of
   filings, low millions of chunks). Past that, swap `storage.py`'s
   `sqlite3` connection for `psycopg2`/`asyncpg` against Postgres — the
   schema and natural-key UPSERT pattern carry over almost unchanged,
   which is the whole point of designing idempotency around natural keys
   instead of auto-increment IDs from the start.
4. **Raw file storage**: swap `data/raw_filings/` (local disk) for S3/GCS
   with the same accession-number-keyed paths once you're past what fits
   on one machine.
5. **Queueing**: for true million-document scale, put filing-discovery
   and filing-processing behind a queue (SQS/Redis/Celery) so downloads
   and parsing scale independently and a crashed worker just leaves a
   message to be retried, rather than losing an in-memory batch.

None of this requires touching `section_parser.py`, `chunker.py`, or
`xbrl_fetcher.py` — they're pure functions over bytes/text in, structured
rows out, which is what makes the scaling path additive rather than a
rewrite.

## Known limitations to be upfront about

- `section_parser.py` is regex/heading based and will occasionally
  mis-split unusually formatted filings (older filers, non-standard HTML).
  That's *why* the quarantine path exists — expect some quarantine rate
  and treat it as a real metric to report, not a bug to hide.
- Token counts use a whitespace-word proxy, not a real tokenizer. Fine for
  chunk-sizing; swap in `tiktoken` (or your embedding model's tokenizer)
  if you need exact counts later.
- CIK resolution downloads the full SEC ticker map on every call; fine at
  this scale, cache it to disk if you're resolving thousands of tickers.