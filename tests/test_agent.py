"""
Offline tests for Phase 4: Agentic routing (LangGraph).

Tests verify node logic and graph edge decisions using mock LLM
responses.  No real Gemini calls or model downloads occur.

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
    generate_answer,
    grade_answer,
    route_question,
    sql_search,
    vector_search,
)


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

def _mock_client(json_response: dict) -> MagicMock:
    """Creates a mock genai.Client that returns a fixed JSON response."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(json_response)
    mock.models.generate_content.return_value = mock_response
    return mock


# ---------------------------------------------------------------------------
# Tests: Router node
# ---------------------------------------------------------------------------

def test_router_classifies_vector_search():
    client = _mock_client({"route": "vector_search"})
    result = route_question({"question": "What are Apple's risk factors?"}, client)
    assert result["route"] == "vector_search"


def test_router_classifies_sql_search():
    client = _mock_client({"route": "sql_search"})
    result = route_question({"question": "What was AAPL revenue in 2024?"}, client)
    assert result["route"] == "sql_search"


def test_router_classifies_out_of_scope():
    client = _mock_client({"route": "out_of_scope"})
    result = route_question({"question": "What is the weather today?"}, client)
    assert result["route"] == "out_of_scope"


# ---------------------------------------------------------------------------
# Tests: SQL search node
# ---------------------------------------------------------------------------

def test_sql_search_returns_facts():
    """SQL search returns facts when ticker and concept match."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        storage = Storage(db_path)
        storage.upsert_company(cik="0000320193", ticker="AAPL")
        storage.upsert_xbrl_facts([
            ("f1", "0000320193", "NetIncomeLoss", "USD", 94000000000,
             "2023-10-01", "2024-09-28", 2024, "FY", "10-K", "acc-1"),
        ])

        client = _mock_client({"ticker": "AAPL", "concept": "NetIncomeLoss", "fiscal_year": 2024})
        result = sql_search({"question": "What was Apple's net income in 2024?"}, client, storage)
        storage.close()

        assert len(result["sql_results"]) == 1
        assert result["sql_results"][0]["value"] == 94000000000


def test_sql_search_missing_ticker_returns_empty():
    """SQL search with unrecognized ticker returns empty list."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        storage = Storage(db_path)
        client = _mock_client({"ticker": "UNKNOWN", "concept": "Revenues", "fiscal_year": 2024})
        result = sql_search({"question": "What was XYZ's revenue?"}, client, storage)
        storage.close()

        assert result["sql_results"] == []


# ---------------------------------------------------------------------------
# Tests: Grader node
# ---------------------------------------------------------------------------

def test_grader_pass():
    client = _mock_client({"pass": True})
    state = {"question": "test?", "generation": "The answer.", "retries": 0}
    result = grade_answer(state, client)
    assert result["route"] == "pass"


def test_grader_fail_triggers_retry():
    client = _mock_client({"pass": False})
    state = {"question": "test?", "generation": "Bad answer.", "retries": 0}
    result = grade_answer(state, client)
    assert result["route"] == "retry"
    assert result["retries"] == 1


def test_grader_fail_at_cap_returns_insufficient():
    """When retries hit MAX_RETRIES, grader should stop retrying."""
    client = _mock_client({"pass": False})
    state = {"question": "test?", "generation": "Bad answer.", "retries": MAX_RETRIES}
    result = grade_answer(state, client)
    assert result["route"] == "fail_cap"
    assert "don't have sufficient" in result["generation"].lower()


# ---------------------------------------------------------------------------
# Tests: Generate node
# ---------------------------------------------------------------------------

def test_generate_with_no_context():
    """Generate should return insufficient info message when no docs/facts."""
    state = {"question": "test?", "documents": [], "sql_results": []}
    result = generate_answer(state, _mock_client({}))
    assert "don't have sufficient" in result["generation"].lower()


def test_generate_formats_sql_facts_in_context():
    """Generate should include SQL facts in the prompt context."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Net income was $94B."
    client.models.generate_content.return_value = mock_response

    state = {
        "question": "What was Apple's net income?",
        "documents": [],
        "sql_results": [{"concept": "NetIncomeLoss", "value": 94000000000,
                         "unit": "USD", "fiscal_year": 2024, "fiscal_period": "FY"}],
    }
    result = generate_answer(state, client)
    assert result["generation"] == "Net income was $94B."

    # Verify the prompt sent to Gemini included the XBRL fact
    call_args = client.models.generate_content.call_args
    prompt = call_args.kwargs.get("contents") or call_args[1].get("contents", "")
    assert "NetIncomeLoss" in prompt
