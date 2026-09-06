"""
LLM Generation component for the RAG pipelines.
Handles formatting context and calling the Groq API.
"""
from __future__ import annotations

import logging
import os
import time

from groq import Groq

from .schema import RetrievedChunk

logger = logging.getLogger("filingsagent.rag.generation")

DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """\
You are a financial analyst assistant.  You answer questions about SEC 10-K \
filings using ONLY the retrieved context provided below.  If the context does \
not contain enough information to answer the question, say "I don't have \
sufficient information in the retrieved filings to answer this question."

Rules:
- Base every claim strictly on the provided context.
- Cite the source chunk when making a specific claim (use the chunk_id).
- Do not speculate or use information not present in the context.
- Be concise and precise.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Formats retrieved chunks into a context block for the LLM prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        header = (
            f"[Chunk {i}] chunk_id={c.chunk_id[:12]}... | "
            f"ticker={c.ticker} | section={c.section_name} | "
            f"filing_date={c.filing_date}"
        )
        parts.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def generate(
    question: str,
    chunks: list[RetrievedChunk],
    model_name: str | None = None,
) -> str:
    """Calls Groq to generate an answer grounded in retrieved chunks."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GROQ_API_KEY environment variable. "
            "Get a free key at https://console.groq.com/keys"
        )

    model = model_name or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    client = Groq(api_key=api_key)
    context = _format_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return chat_completion.choices[0].message.content or ""
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait_time = 15 if attempt == 0 else 25
                logger.warning(
                    "Groq API rate limit (429) hit in generation. Waiting %ds before retry (%d/%d)...",
                    wait_time, attempt + 1, max_retries
                )
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError("Max retries exceeded due to Groq rate limits.")
