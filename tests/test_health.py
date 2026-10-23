"""
Test suite for Orient BlackSwan History Catalogue RAG Bot API endpoints.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /api/health returns 200 and expected health indicators."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "Orient BlackSwan History Catalogue RAG Bot"
    assert "checks" in data
    assert "catalogue_pdf_present" in data["checks"]
    assert "vector_database_indexed" in data["checks"]
    assert "gemini_api_key_configured" in data["checks"]


def test_root_serves_html():
    """Verify GET / serves the frontend HTML interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Orient BlackSwan History Catalogue" in response.text


def test_query_endpoint_schema():
    """Verify POST /api/query returns structured response matching QueryResponse schema."""
    test_question = "In which year was Orient BlackSwan incorporated?"
    response = client.post("/api/query", json={"question": test_question})
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == test_question
    assert "answer" in data
    assert "citations" in data
    assert isinstance(data["citations"], list)
    assert isinstance(data["found"], bool)
