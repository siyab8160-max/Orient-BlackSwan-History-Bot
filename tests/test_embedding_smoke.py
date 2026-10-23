"""
Live API Smoke Test for Google Gemini Embeddings
================================================
Minimal, isolated smoke test that verifies live connectivity to the Google Gemini API.
Specifically tests:
1. Generating a document embedding for a single short sentence.
2. Generating a query embedding for a single short query.
3. Confirming that both use the exact same model and produce identical, compatible dimensions.
4. Does NOT index the entire document (avoids running all 750 chunks).
"""

import os
import sys
from pathlib import Path
import pytest
from dotenv import load_dotenv

# Ensure app is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

load_dotenv()

from app.ingest import get_gemini_api_key, DEFAULT_EMBEDDING_MODEL
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def run_embedding_smoke_test(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    Execute minimal live API smoke test for one document sentence and one query.
    Returns a dictionary of test metrics.
    """
    try:
        api_key = get_gemini_api_key()
    except ValueError as e:
        pytest.skip(f"Live Gemini API key not configured: {e}")

    sample_sentence = "Orient BlackSwan History Catalogue 2024-25 provides academic bibliographic listings."
    sample_query = "What monographs on Mughal administration and agrarian systems are listed?"

    print(f"\n[SMOKE TEST] Initializing GoogleGenerativeAIEmbeddings with model: '{model_name}'...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key,
    )

    # 1. Embed single document sentence
    print(f"[SMOKE TEST] Calling embed_documents with 1 sentence: \"{sample_sentence[:45]}...\"")
    doc_vectors = embeddings.embed_documents([sample_sentence])
    assert len(doc_vectors) == 1, "Expected exactly 1 document embedding"
    doc_vector = doc_vectors[0]
    assert len(doc_vector) > 0, "Document vector should not be empty"

    # 2. Embed single query
    print(f"[SMOKE TEST] Calling embed_query with 1 query: \"{sample_query[:45]}...\"")
    query_vector = embeddings.embed_query(sample_query)
    assert len(query_vector) > 0, "Query vector should not be empty"

    # 3. Verify dimension compatibility
    doc_dim = len(doc_vector)
    query_dim = len(query_vector)
    assert doc_dim == query_dim, (
        f"Dimension mismatch! Document embedding dimension is {doc_dim}, "
        f"but query embedding dimension is {query_dim}."
    )

    # 4. Verify vector contents
    assert all(isinstance(v, float) for v in doc_vector[:10]), "Document vector should contain float values"
    assert all(isinstance(v, float) for v in query_vector[:10]), "Query vector should contain float values"

    return {
        "status": "PASSED",
        "model": model_name,
        "document_sentence": sample_sentence,
        "query": sample_query,
        "doc_embedding_dimension": doc_dim,
        "query_embedding_dimension": query_dim,
        "dimension_compatible": doc_dim == query_dim,
        "doc_vector_sample": doc_vector[:5],
        "query_vector_sample": query_vector[:5],
    }


def test_live_gemini_embedding_smoke():
    """Pytest test case for live Gemini embedding smoke test."""
    results = run_embedding_smoke_test()
    assert results["status"] == "PASSED"
    assert results["dimension_compatible"] is True
    assert results["doc_embedding_dimension"] == 3072


if __name__ == "__main__":
    print("=" * 70)
    print(" Google Gemini Embedding Minimal Live Smoke Test")
    print("=" * 70)
    try:
        report = run_embedding_smoke_test()
        print("\n[RESULT] Live API Smoke Test Passed Successfully!")
        print(f" - Model Tested:                {report['model']}")
        print(f" - Document Vector Dimensions:  {report['doc_embedding_dimension']}")
        print(f" - Query Vector Dimensions:     {report['query_embedding_dimension']}")
        print(f" - Dimensions Compatible:       {report['dimension_compatible']}")
        print(f" - Document Sample Floats:      {[round(x, 6) for x in report['doc_vector_sample']]}")
        print(f" - Query Sample Floats:         {[round(x, 6) for x in report['query_vector_sample']]}")
        print("=" * 70)
    except Exception as exc:
        print(f"\n[FAILED] Live API Smoke Test failed: {exc}", file=sys.stderr)
        sys.exit(1)
