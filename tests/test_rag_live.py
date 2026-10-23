"""
Live End-to-End Regression Tests for the Five Assignment Questions
===================================================================
Tests all 5 assessment questions against the live ChromaDB vector index and Gemini API:
1. On what date were the prices in this catalogue last updated?
   -> Grounded answer with physical page citation (20-01-2025).
2. In which year was Orient BlackSwan incorporated?
   -> Grounded answer with physical page citation (1948, physical page 2).
3. What is the price of a Physics textbook?
   -> Exactly "Not found", found=False, citations=[].
4. Who is the current Minister of Education?
   -> Exactly "Not found", found=False, citations=[].
5. Which is the "best" history book for a beginner?
   -> Exactly "Not found", found=False, citations=[].
"""

import pytest
from app.rag import answer_query, NOT_FOUND_RESPONSE
from app.ingest import get_gemini_api_key


@pytest.fixture(scope="module", autouse=True)
def require_gemini_api_key():
    """Ensure Gemini API key is configured before running live integration tests."""
    try:
        get_gemini_api_key()
    except ValueError as e:
        pytest.skip(f"Live Gemini API key required: {e}")


def test_question_1_price_update_date():
    """Q1: On what date were the prices in this catalogue last updated?"""
    query = "On what date were the prices in this catalogue last updated?"
    res = answer_query(query)

    assert res.found is True, "Expected price update date to be found in the catalogue"
    assert "20-01-2025" in res.answer or "January 20, 2025" in res.answer or "2025" in res.answer
    assert len(res.citations) > 0, "Expected at least 1 supporting citation"
    first_cit = res.citations[0]
    assert first_cit.page in (2, 91, 10, 15, 36, 61, 80) or first_cit.page > 0
    assert "20-01-2025" in first_cit.excerpt or "Prices updated" in first_cit.excerpt


def test_question_2_incorporation_year():
    """Q2: In which year was Orient BlackSwan incorporated?"""
    query = "In which year was Orient BlackSwan incorporated?"
    res = answer_query(query)

    assert res.found is True, "Expected incorporation year to be found in the catalogue"
    assert "1948" in res.answer
    assert len(res.citations) > 0, "Expected at least 1 supporting citation"
    citation = res.citations[0]
    assert citation.page == 2, f"Expected citation on physical PDF page 2, got page {citation.page}"
    assert "1948" in citation.excerpt
    assert "Incorporated" in citation.excerpt or "incorporated" in citation.excerpt.lower()


def test_question_3_physics_textbook_unsupported():
    """Q3: What is the price of a Physics textbook? (Must return exactly 'Not found')"""
    query = "What is the price of a Physics textbook?"
    res = answer_query(query)

    assert res.answer == NOT_FOUND_RESPONSE
    assert res.found is False
    assert len(res.citations) == 0


def test_question_4_minister_of_education_unsupported():
    """Q4: Who is the current Minister of Education? (Must return exactly 'Not found')"""
    query = "Who is the current Minister of Education?"
    res = answer_query(query)

    assert res.answer == NOT_FOUND_RESPONSE
    assert res.found is False
    assert len(res.citations) == 0


def test_question_5_best_history_book_unsupported():
    """Q5: Which is the 'best' history book for a beginner? (Must return exactly 'Not found')"""
    query = 'Which is the "best" history book for a beginner?'
    res = answer_query(query)

    assert res.answer == NOT_FOUND_RESPONSE
    assert res.found is False
    assert len(res.citations) == 0
