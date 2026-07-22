"""
Splits a raw 10-K HTML document into the canonical sections defined in
config.TARGET_SECTIONS (Item 1, 1A, 7, 7A, 8).

Why section-first: chunking a 10-K with flat fixed-size windows mixes
boilerplate legal text, risk factors, and financial commentary into the
same chunks, which hurts retrieval precision badly. Splitting on SEC's own
"Item N." headings first, and only THEN doing token-based chunking inside
each section, keeps each chunk topically coherent and lets us filter
retrieval by section (e.g. "only search Risk Factors").

This is a best-effort text-based parser (real 10-Ks have wildly
inconsistent HTML), so it is intentionally defensive: if fewer than 2
sections are found, the caller should quarantine the document rather than
silently ingesting an unparsed blob.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from .config import TARGET_SECTIONS


@dataclass
class Section:
    name: str
    text: str


def html_to_clean_text(raw_html: bytes) -> str:
    """Strips tags/scripts/styles, collapses whitespace, keeps paragraph breaks."""
    soup = BeautifulSoup(raw_html, features="xml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    # get_text with a separator preserves block-level breaks so headings
    # don't get glued to the previous paragraph.
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


# A 10-K's table of contents also contains lines like "Item 1A. Risk
# Factors ... 14" -- we need the SECOND occurrence (the actual section
# body), not the ToC entry. We handle this by taking the LAST match of
# each pattern that is followed by a large amount of subsequent text,
# which in practice reliably skips short ToC lines.
_HEADER_RE_TEMPLATE = r"(?im)^\s*item\s+{num}\.?\s+{title}"


def split_into_sections(clean_text: str) -> list[Section]:
    matches = []  # (start_offset, canonical_name)
    for canonical_name, pattern in TARGET_SECTIONS.items():
        for m in re.finditer(pattern, clean_text, flags=re.IGNORECASE):
            matches.append((m.start(), canonical_name))

    if not matches:
        return []

    matches.sort(key=lambda t: t[0])

    # De-duplicate near-identical headers (ToC vs. real section): if the
    # same canonical_name appears more than once, keep only the occurrence
    # that has the most text before the NEXT different-section header --
    # i.e. the one that actually behaves like a section body, not a ToC
    # line sitting a few characters above the next ToC line.
    best_by_name: dict[str, tuple[int, int]] = {}
    for i, (start, name) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(clean_text)
        span = next_start - start
        if name not in best_by_name or span > best_by_name[name][1]:
            best_by_name[name] = (start, span)

    ordered = sorted(best_by_name.items(), key=lambda kv: kv[1][0])
    sections = []
    for idx, (name, (start, _span)) in enumerate(ordered):
        end = ordered[idx + 1][1][0] if idx + 1 < len(ordered) else len(clean_text)
        body = clean_text[start:end].strip()
        if body:
            sections.append(Section(name=name, text=body))
    return sections
