"""
LLM Generation component for the RAG pipelines.
Handles formatting context and calling the Gemini API.
"""
from __future__ import annotations

import os

from google import genai

from .schema import RetrievedChunk

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
    model_name: str = "gemini-2.0-flash",
) -> str:
    """Calls Gemini to generate an answer grounded in retrieved chunks."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)
    context = _format_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        ),
    )
    return response.text
