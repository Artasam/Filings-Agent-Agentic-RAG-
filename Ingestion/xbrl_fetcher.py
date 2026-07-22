"""
Fetches structured financial facts (XBRL) for a company and normalizes them
into flat rows ready for SQL storage -- this is what powers the
"quantitative" retrieval path in the agent (revenue, EPS, etc. looked up
directly rather than retrieved via embedding search).

SEC's companyfacts payload is deeply nested:
  facts -> "us-gaap" -> {concept} -> "units" -> {unit} -> [ {val, start, end,
  fy, fp, form, accn, ...}, ... ]
We flatten this into one row per (concept, period, unit) fact.
"""
from __future__ import annotations

import hashlib
from typing import Iterator

from .config import XBRL_CONCEPTS
from .storage import make_id


def extract_facts(cik: str, company_facts_json: dict) -> Iterator[tuple]:
    """
    Yields rows shaped for Storage.upsert_xbrl_facts:
    (fact_id, cik, concept, unit, value, period_start, period_end,
     fiscal_year, fiscal_period, form, accession_no)
    """
    us_gaap = company_facts_json.get("facts", {}).get("us-gaap", {})
    for concept in XBRL_CONCEPTS:
        concept_data = us_gaap.get(concept)
        if not concept_data:
            continue
        units = concept_data.get("units", {})
        for unit, entries in units.items():
            for entry in entries:
                value = entry.get("val")
                if value is None:
                    continue
                period_start = entry.get("start", "")
                period_end = entry.get("end", "")
                fiscal_year = entry.get("fy")
                fiscal_period = entry.get("fp")
                form = entry.get("form")
                accn = entry.get("accn")
                fact_id = make_id(cik, concept, unit, period_start, period_end, str(value), str(accn))
                yield (
                    fact_id, cik, concept, unit, float(value),
                    period_start, period_end, fiscal_year, fiscal_period, form, accn,
                )
