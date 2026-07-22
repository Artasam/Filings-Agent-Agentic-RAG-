"""
Section-aware, token-budgeted chunker.

Strategy: within each section (already isolated by section_parser), split
on paragraph boundaries first, then greedily pack paragraphs into chunks
up to `target_chunk_tokens`, carrying the trailing `chunk_overlap_tokens`
of the previous chunk into the next one. A paragraph longer than the
target on its own is hard-split on sentence boundaries as a fallback.

Token counting: this project uses a whitespace-word count as a token proxy
(no tokenizer dependency required). It's ~15-20% off from a real BPE
tokenizer's count, which is fine for chunk-sizing purposes -- swap in
`tiktoken` if you want exact counts for your specific embedding model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def count_tokens(text: str) -> int:
    return len(text.split())


@dataclass
class Chunk:
    section_name: str
    chunk_index: int
    text: str
    token_count: int


def _split_paragraphs(section_text: str) -> list[str]:
    paras = [p.strip() for p in section_text.split("\n") if p.strip()]
    return paras


def _hard_split_long_paragraph(paragraph: str, target_tokens: int) -> list[str]:
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)
    pieces, current, current_tokens = [], [], 0
    for sent in sentences:
        t = count_tokens(sent)
        if current and current_tokens + t > target_tokens:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sent)
        current_tokens += t
    if current:
        pieces.append(" ".join(current))
    return pieces


def _take_overlap_tail(text: str, overlap_tokens: int) -> str:
    words = text.split()
    if len(words) <= overlap_tokens:
        return text
    return " ".join(words[-overlap_tokens:])


def chunk_section(
    section_name: str,
    section_text: str,
    target_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
) -> list[Chunk]:
    paragraphs = _split_paragraphs(section_text)

    # Expand any paragraph that alone exceeds the target so the greedy
    # packer below never has to handle an oversized unit.
    units: list[str] = []
    for p in paragraphs:
        if count_tokens(p) > target_tokens:
            units.extend(_hard_split_long_paragraph(p, target_tokens))
        else:
            units.append(p)

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    chunk_index = 0

    def flush(carry_overlap: bool) -> None:
        nonlocal current_parts, current_tokens, chunk_index
        if not current_parts:
            return
        text = "\n".join(current_parts)
        chunks.append(
            Chunk(section_name=section_name, chunk_index=chunk_index, text=text, token_count=count_tokens(text))
        )
        chunk_index += 1
        if carry_overlap:
            tail = _take_overlap_tail(text, overlap_tokens)
            current_parts = [tail]
            current_tokens = count_tokens(tail)
        else:
            current_parts = []
            current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        if current_tokens + unit_tokens > target_tokens and current_parts:
            flush(carry_overlap=True)
        current_parts.append(unit)
        current_tokens += unit_tokens

    flush(carry_overlap=False)

    # Merge a trailing tiny fragment into the previous chunk rather than
    # storing a near-empty, low-signal chunk.
    if len(chunks) >= 2 and chunks[-1].token_count < min_tokens:
        last = chunks.pop()
        prev = chunks.pop()
        merged_text = prev.text + "\n" + last.text
        chunks.append(
            Chunk(section_name=section_name, chunk_index=prev.chunk_index,
                  text=merged_text, token_count=count_tokens(merged_text))
        )

    return chunks
