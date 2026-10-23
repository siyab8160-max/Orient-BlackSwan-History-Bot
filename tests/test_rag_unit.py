"""
Unit tests for app/rag.py logic without live API dependencies.
Verifies:
1. clean_llm_json_text handles string, list, and markdown code fences.
2. validate_citations_against_chunks verifies physical page and verbatim excerpt matching.
3. Invalid or hallucinated citations are stripped.
4. Schema validation for QueryRequest, Citation, and QueryResponse.
"""

import pytest
from langchain_core.documents import Document
from app.rag import (
    clean_llm_json_text,
    validate_citations_against_chunks,
    Citation,
    QueryRequest,
    QueryResponse,
    NOT_FOUND_RESPONSE,
)


def test_clean_llm_json_text():
    """Verify markdown code fence stripping from raw LLM output."""
    raw_markdown = '```json\n{"answer": "Test answer", "citations": [], "found": true}\n```'
    cleaned = clean_llm_json_text(raw_markdown)
    assert cleaned == '{"answer": "Test answer", "citations": [], "found": true}'

    raw_fence_no_json = '```\n{"answer": "Not found"}\n```'
    assert clean_llm_json_text(raw_fence_no_json) == '{"answer": "Not found"}'

    list_input = [{"type": "text", "text": '{"answer": "Valid text"}'}]
    assert clean_llm_json_text(list_input) == '{"answer": "Valid text"}'


def test_validate_citations_valid_matching():
    """Verify that citations matching retrieved chunks are kept with 1-indexed page numbers."""
    retrieved_chunks = [
        Document(
            page_content="Orient BlackSwan is one of India's best known publishing houses. Incorporated in 1948.",
            metadata={"page": 1, "source": "History_2024-25.pdf"},  # zero-based index 1 -> physical page 2
        ),
        Document(
            page_content="Prices updated on 20-01-2025. These prices are subject to change without notice.",
            metadata={"page": 90, "source": "History_2024-25.pdf"},  # zero-based index 90 -> physical page 91
        ),
    ]

    raw_citations = [
        {"page": 2, "excerpt": "Incorporated in 1948"},
        {"page": 91, "excerpt": "Prices updated on 20-01-2025"},
    ]

    validated = validate_citations_against_chunks(raw_citations, retrieved_chunks)
    assert len(validated) == 2
    assert validated[0].page == 2
    assert validated[0].excerpt == "Incorporated in 1948"
    assert validated[1].page == 91
    assert validated[1].excerpt == "Prices updated on 20-01-2025"


def test_validate_citations_rejects_hallucinations():
    """Verify that citations not supported by retrieved chunks are discarded."""
    retrieved_chunks = [
        Document(
            page_content="Monographs on Early Medieval Indian society and trade guilds.",
            metadata={"page": 10, "source": "History_2024-25.pdf"},
        ),
    ]

    hallucinated_citations = [
        {"page": 5, "excerpt": "This sentence does not exist anywhere in the text."},
        {"page": 11, "excerpt": "Completely invented excerpt"},
        {"page": -1, "excerpt": ""},
    ]

    validated = validate_citations_against_chunks(hallucinated_citations, retrieved_chunks)
    assert len(validated) == 0


def test_query_schema_validation():
    """Verify Pydantic models for request and response."""
    req = QueryRequest(question="Where was the catalogue printed?")
    assert req.question == "Where was the catalogue printed?"

    with pytest.raises(Exception):
        QueryRequest(question="")

    resp = QueryResponse(
        question="What is the title?",
        answer=NOT_FOUND_RESPONSE,
        citations=[],
        found=False,
    )
    assert resp.found is False
    assert resp.answer == "Not found"
    assert resp.citations == []
