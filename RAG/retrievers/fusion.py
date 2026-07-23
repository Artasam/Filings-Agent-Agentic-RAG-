"""
Reciprocal Rank Fusion.
"""
from __future__ import annotations

from ..schema import RetrievedChunk


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Merges multiple ranked lists using RRF.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            rrf_scores[chunk.chunk_id] = (
                rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            )
            # Keep first occurrence (preserves metadata from higher-ranked list)
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    return [
        RetrievedChunk(
            chunk_id=cid,
            text=chunk_map[cid].text,
            score=rrf_scores[cid],
            ticker=chunk_map[cid].ticker,
            section_name=chunk_map[cid].section_name,
            accession_no=chunk_map[cid].accession_no,
            filing_date=chunk_map[cid].filing_date,
            fiscal_year=chunk_map[cid].fiscal_year,
        )
        for cid in sorted_ids
    ]
