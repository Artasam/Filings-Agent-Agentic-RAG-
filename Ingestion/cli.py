"""
CLI entrypoint.

Usage:
    python -m filingsagent.cli ingest --tickers AAPL MSFT GOOGL --forms 10-K --per-company 3
    python -m filingsagent.cli seed-eval --n 30
"""
from __future__ import annotations

import argparse
import dataclasses

from .config import CONFIG
from .eval_seed import write_eval_seed
from .guardrails import logger
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="FilingsAgent ingestion pipeline (Phase 1)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Download + parse + chunk filings, fetch XBRL facts")
    p_ingest.add_argument("--tickers", nargs="+", required=True, help="e.g. AAPL MSFT GOOGL")
    p_ingest.add_argument("--forms", nargs="+", default=list(CONFIG.forms_to_ingest))
    p_ingest.add_argument("--per-company", type=int, default=CONFIG.filings_per_company)
    p_ingest.add_argument("--user-agent", default=CONFIG.user_agent,
                           help="REQUIRED by SEC: 'Your Name your_email@example.com'")
    p_ingest.add_argument("--workers", type=int, default=CONFIG.max_workers)
    p_ingest.add_argument("--rps", type=float, default=CONFIG.max_requests_per_second)

    p_seed = sub.add_parser("seed-eval", help="Sample stored chunks/XBRL facts into an eval question-bank template")
    p_seed.add_argument("--n", type=int, default=40, help="number of candidate seeds to sample")
    p_seed.add_argument("--out", default="data/eval_seed.csv")

    args = parser.parse_args()

    if args.command == "ingest":
        config = dataclasses.replace(
            CONFIG,
            forms_to_ingest=tuple(args.forms),
            filings_per_company=args.per_company,
            user_agent=args.user_agent,
            max_workers=args.workers,
            max_requests_per_second=args.rps,
        )
        if "example.com" in config.user_agent:
            logger.warning(
                "You are using the placeholder User-Agent. SEC will likely block you -- "
                "pass --user-agent 'Your Name your_real_email@domain.com'."
            )
        run(config, args.tickers)

    elif args.command == "seed-eval":
        write_eval_seed(CONFIG, n=args.n, out_path=args.out)


if __name__ == "__main__":
    main()
