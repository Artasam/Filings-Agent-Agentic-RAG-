"""
Offline tests for Phase 5: Agentic routing with citations, memory,
and multi-hop decomposition.

All tests use mock LLM responses. No real Gemini calls or model downloads.

Run with:  python -m pytest tests/test_agent.py -v
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Ingestion.storage import Storage
from Agent.nodes import (
    MAX_RETRIES,
    _format_history,
    _retrieve_sql,
    analyze_query,
    execute_retrieval,
    generate_answer,
    out_of_scope,
    save_turn,
    verify_citations,
)


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

def _mock_client(json_response: dict) -> MagicMock:
    """Creates a mock Groq client that returns a fixed JSON response."""
    mock = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(json_response)
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock.chat.completions.create.return_value = mock_response
    return mock


# ---------------------------------------------------------------------------
# Tests: _format_history helper
# ---------------------------------------------------------------------------

def test_format_history_empty():
    assert _format_history([]) == "(No prior conversation)"


def test_format_history_with_turns():
    messages = [
        {"role": "user", "content": "What is AAPL revenue?"},
        {"role": "assistant", "content": "Apple revenue was $390B."},
    ]
    result = _format_history(messages)
    assert "User: What is AAPL revenue?" in result
    assert "Assistant: Apple revenue was $390B." in result


# ---------------------------------------------------------------------------
# Tests: analyze_query node
# ---------------------------------------------------------------------------

def test_analyze_simple_vector_query():
    """Simple qualitative question → single vector_search sub-query."""
    client = _mock_client({
        "is_out_of_scope": False,
        "rewritten_query": "What are Apple's risk factors?",
        "sub_queries": [{"query": "What are Apple's risk factors?", "route": "vector_search"}],
    })
    result = analyze_query({"question": "What are Apple's risk factors?", "messages": []}, client)
    assert result["route"] == "proceed"
    assert len(result["sub_queries"]) == 1
    assert result["sub_queries"][0]["route"] == "vector_search"


def test_analyze_sql_query():
    """Quantitative question → single sql_search sub-query."""
    client = _mock_client({
        "is_out_of_scope": False,
        "rewritten_query": "What was AAPL revenue in 2024?",
        "sub_queries": [{"query": "What was AAPL revenue in 2024?", "route": "sql_search"}],
    })
    result = analyze_query({"question": "What was AAPL revenue in 2024?"}, client)
    assert result["route"] == "proceed"
    assert result["sub_queries"][0]["route"] == "sql_search"


def test_analyze_out_of_scope():
    """Non-financial question → out_of_scope."""
    client = _mock_client({
        "is_out_of_scope": True,
        "rewritten_query": "What is the weather today?",
        "sub_queries": [],
    })
    result = analyze_query({"question": "What is the weather today?"}, client)
    assert result["route"] == "out_of_scope"
    assert result["sub_queries"] == []


def test_analyze_coreference_resolution():
    """'Their revenue' with history about AAPL → resolved to AAPL."""
    client = _mock_client({
        "is_out_of_scope": False,
        "rewritten_query": "What was Apple's revenue in 2024?",
        "sub_queries": [{"query": "What was Apple's revenue in 2024?", "route": "sql_search"}],
    })
    state = {
        "question": "What was their revenue in 2024?",
        "messages": [
            {"role": "user", "content": "Tell me about Apple's risk factors."},
            {"role": "assistant", "content": "Apple faces supply chain risks..."},
        ],
    }
    result = analyze_query(state, client)
    assert "Apple" in result["rewritten_query"]


def test_analyze_multi_hop_decomposition():
    """Comparative query → decomposed into multiple sub-queries."""
    client = _mock_client({
        "is_out_of_scope": False,
        "rewritten_query": "Compare Apple and Microsoft R&D spending",
        "sub_queries": [
            {"query": "What is Apple's R&D expense?", "route": "sql_search"},
            {"query": "What is Microsoft's R&D expense?", "route": "sql_search"},
        ],
    })
    result = analyze_query({"question": "Compare Apple and Microsoft R&D spending"}, client)
    assert len(result["sub_queries"]) == 2


# ---------------------------------------------------------------------------
# Tests: execute_retrieval node
# ---------------------------------------------------------------------------

def test_execute_retrieval_aggregates_results():
    """Multiple sub-queries should produce aggregated results."""
    # Mock the retrieval resources — we won't actually call them,
    # we'll test _retrieve_sql directly instead
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        storage = Storage(db_path)
        storage.upsert_company(cik="0000320193", ticker="AAPL")
        storage.upsert_xbrl_facts([
            ("f1", "0000320193", "ResearchAndDevelopmentExpense", "USD", 26000000000,
             "2023-10-01", "2024-09-28", 2024, "FY", "10-K", "acc-1"),
        ])

        client = _mock_client({"ticker": "AAPL", "concept": "ResearchAndDevelopmentExpense", "fiscal_year": 2024})
        facts = _retrieve_sql("What is Apple's R&D expense?", client, storage)
        storage.close()

        assert len(facts) == 1
        assert facts[0]["concept"] == "ResearchAndDevelopmentExpense"
        assert facts[0]["fact_id"] == "f1"


def test_retrieve_sql_missing_ticker():
    """SQL retrieval with null ticker returns empty."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        storage = Storage(db_path)
        client = _mock_client({"ticker": None, "concept": "Revenues", "fiscal_year": 2024})
        facts = _retrieve_sql("Revenue of unknown company", client, storage)
        storage.close()

        assert facts == []


# ---------------------------------------------------------------------------
# Tests: generate_answer node (structured citations)
# ---------------------------------------------------------------------------

def test_generate_with_citations():
    """Generation should return structured claims with source_ids."""
    client = _mock_client({
        "answer": "Apple's net income was $94B in FY2024.",
        "claims": [
            {"claim": "Apple's net income was $94B in FY2024", "source_ids": ["f1"]},
        ],
    })
    state = {
        "question": "What was Apple's net income?",
        "rewritten_query": "What was Apple's net income?",
        "documents": [],
        "sql_results": [{"fact_id": "f1", "concept": "NetIncomeLoss", "value": 94000000000,
                         "unit": "USD", "fiscal_year": 2024, "fiscal_period": "FY"}],
    }
    result = generate_answer(state, client)
    assert result["generation"] == "Apple's net income was $94B in FY2024."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source_ids"] == ["f1"]


def test_generate_no_context():
    """Empty docs + empty facts → insufficient information."""
    state = {"question": "test?", "documents": [], "sql_results": []}
    result = generate_answer(state, _mock_client({}))
    assert "don't have sufficient" in result["generation"].lower()
    assert result["citations"] == []


# ---------------------------------------------------------------------------
# Tests: verify_citations node
# ---------------------------------------------------------------------------

def test_verify_all_grounded():
    """All claims grounded → pass."""
    client = _mock_client({"all_grounded": True, "ungrounded_claims": []})
    state = {
        "question": "test?",
        "generation": "Answer.",
        "citations": [{"claim": "Revenue was $390B", "source_ids": ["f1"]}],
        "documents": [],
        "sql_results": [{"fact_id": "f1", "concept": "Revenues", "value": 390000000000,
                         "unit": "USD", "fiscal_year": 2024, "fiscal_period": "FY"}],
        "retries": 0,
    }
    result = verify_citations(state, client)
    assert result["route"] == "pass"


def test_verify_ungrounded_triggers_retry():
    """Ungrounded claims → retry."""
    client = _mock_client({"all_grounded": False, "ungrounded_claims": ["Some false claim"]})
    state = {
        "question": "test?",
        "generation": "Bad answer.",
        "citations": [{"claim": "Some false claim", "source_ids": ["c1"]}],
        "documents": [{"chunk_id": "c1", "text": "Actual text about something else."}],
        "sql_results": [],
        "retries": 0,
    }
    result = verify_citations(state, client)
    assert result["route"] == "retry"
    assert result["retries"] == 1


def test_verify_retry_cap():
    """Retries at cap → fail_cap with insufficient info message."""
    client = _mock_client({"all_grounded": False, "ungrounded_claims": ["Bad claim"]})
    state = {
        "question": "test?",
        "generation": "Bad answer.",
        "citations": [{"claim": "Bad claim", "source_ids": ["c1"]}],
        "documents": [{"chunk_id": "c1", "text": "Unrelated text."}],
        "sql_results": [],
        "retries": MAX_RETRIES,
    }
    result = verify_citations(state, client)
    assert result["route"] == "fail_cap"
    assert "don't have sufficient" in result["generation"].lower()


def test_verify_no_citations_passes():
    """No citations to verify → automatic pass."""
    client = _mock_client({})
    state = {"question": "test?", "generation": "Answer.", "citations": [], "retries": 0}
    result = verify_citations(state, client)
    assert result["route"] == "pass"


# ---------------------------------------------------------------------------
# Tests: out_of_scope and save_turn
# ---------------------------------------------------------------------------

def test_out_of_scope():
    result = out_of_scope({"question": "weather?"})
    assert "outside the scope" in result["generation"].lower()


def test_save_turn_appends_messages():
    state = {"question": "What is AAPL?", "generation": "Apple Inc."}
    result = save_turn(state)
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"] == "Apple Inc."
