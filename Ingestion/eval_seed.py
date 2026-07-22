"""
Samples stored chunks and XBRL facts into a CSV template you fill in by
hand to build the golden eval set described in the blueprint (Section 4).

This does NOT generate questions for you -- it does the tedious part
(finding a spread of candidate source material across companies, years,
and sections/concepts) so you spend your time writing good questions and
answers instead of hunting for source passages.

Output columns:
    id, category, cik, source_type, source_ref, seed_text, question,
    expected_answer, supporting_ids, notes

`question`, `expected_answer`, `supporting_ids` are left blank for you to
fill in. `category` is pre-filled based on source_type as a starting
suggestion (qualitative / quantitative) -- change it if a seed is actually
better suited to a comparative/multi-hop or out-of-scope question.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

from .config import Config
from .storage import Storage


def write_eval_seed(config: Config, n: int, out_path: str) -> None:
    storage = Storage(config.db_path)
    rows = []

    with storage.cursor() as cur:
        cur.execute(
            "SELECT chunk_id, cik, section_name, accession_no, text FROM chunks ORDER BY RANDOM() LIMIT ?",
            (n // 2,),
        )
        chunk_samples = cur.fetchall()

        cur.execute(
            "SELECT fact_id, cik, concept, unit, value, period_end, fiscal_year, form FROM xbrl_facts "
            "ORDER BY RANDOM() LIMIT ?",
            (n - len(chunk_samples),),
        )
        fact_samples = cur.fetchall()

    idx = 0
    for chunk_id, cik, section_name, accession_no, text in chunk_samples:
        idx += 1
        rows.append({
            "id": f"seed_{idx:04d}",
            "category": "qualitative",
            "cik": cik,
            "source_type": "chunk",
            "source_ref": f"{accession_no}::{section_name}::{chunk_id[:12]}",
            "seed_text": text[:400].replace("\n", " "),
            "question": "",
            "expected_answer": "",
            "supporting_ids": chunk_id,
            "notes": "",
        })

    for fact_id, cik, concept, unit, value, period_end, fiscal_year, form in fact_samples:
        idx += 1
        rows.append({
            "id": f"seed_{idx:04d}",
            "category": "quantitative",
            "cik": cik,
            "source_type": "xbrl_fact",
            "source_ref": f"{concept}::{period_end}::FY{fiscal_year}",
            "seed_text": f"{concept} = {value} {unit} (period end {period_end}, form {form})",
            "question": "",
            "expected_answer": "",
            "supporting_ids": fact_id,
            "notes": "",
        })

    random.shuffle(rows)
    for i, r in enumerate(rows, start=1):
        r["id"] = f"seed_{i:04d}"

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "category", "cik", "source_type", "source_ref",
            "seed_text", "question", "expected_answer", "supporting_ids", "notes",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} eval seeds to {out}")
    print("Next: open the CSV and fill in `question` + `expected_answer` for each row by hand.")
    print("Aim for ~120-150 total across multiple ingestion runs, plus some hand-written")
    print("comparative/multi-hop and out-of-scope questions that have no single seed row.")
