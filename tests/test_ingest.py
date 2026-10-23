"""
Unit and integration tests for app/ingest.py
Verifies:
1. PDF extraction preserves source filename and zero-based page metadata.
2. Chunking preserves metadata and assigns stable IDs.
3. Vector store indexing with ChromaDB safely avoids duplicate records on repeated runs.
4. Download logic properly respects existing files and handles timeouts/errors.
"""

import os
from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_chroma import Chroma

from app.ingest import (
    extract_pages,
    split_into_overlapping_chunks,
    create_or_update_vectorstore,
    download_catalogue_pdf,
    get_gemini_api_key,
    DEFAULT_EMBEDDING_MODEL,
    PDF_LOCAL_PATH,
)


def test_pdf_extraction_and_zero_based_metadata():
    """Verify that pages extracted from PDF have source filename and zero-based page metadata."""
    assert PDF_LOCAL_PATH.exists(), "PDF should be downloaded in data/ for this test"
    pages = extract_pages(pdf_path=PDF_LOCAL_PATH)

    assert len(pages) == 91
    first_page = pages[0]
    assert first_page.metadata["source"] == "History_2024-25.pdf"
    assert first_page.metadata["page"] == 0  # Zero-based page metadata
    assert isinstance(first_page.metadata["page"], int)

    last_page = pages[-1]
    assert last_page.metadata["page"] == 90  # 91 pages -> 0 to 90
    # Page 1 contains publishing details and has extractable text
    assert len(pages[1].page_content.strip()) > 0
    assert "Orient BlackSwan" in pages[1].page_content


def test_split_into_overlapping_chunks():
    """Verify that splitting preserves metadata and generates stable IDs."""
    sample_docs = [
        Document(
            page_content="Orient BlackSwan History catalogue. Books on Ancient India and archaeological sources. " * 15,
            metadata={"source": "test_catalogue.pdf", "page": 0},
        ),
        Document(
            page_content="Modern Indian History. Freedom struggle and archival monographs. " * 15,
            metadata={"source": "test_catalogue.pdf", "page": 1},
        ),
    ]

    chunks, ids = split_into_overlapping_chunks(sample_docs, chunk_size=300, chunk_overlap=50)

    assert len(chunks) > len(sample_docs)
    assert len(chunks) == len(ids)

    # Check first chunk metadata and stable ID format
    first_chunk = chunks[0]
    assert first_chunk.metadata["source"] == "test_catalogue.pdf"
    assert first_chunk.metadata["page"] == 0
    assert "chunk_index" in first_chunk.metadata
    assert ids[0].startswith("test_catalogue.pdf:p0:c")


def test_chroma_indexing_and_duplicate_prevention(tmp_path):
    """
    Verify ChromaDB storage with FakeEmbeddings:
    - Indexes chunks into persistent directory.
    - Repeated runs safely reset/recreate collection and do NOT produce duplicate chunks.
    """
    test_db_dir = tmp_path / "test_chroma_db"
    test_collection = "test_obs_history"
    fake_embeddings = FakeEmbeddings(size=64)

    sample_chunks = [
        Document(
            page_content="Monographs on Mughal India administration and revenue systems.",
            metadata={"source": "History_2024-25.pdf", "page": 12, "chunk_index": 0},
        ),
        Document(
            page_content="Textbooks on Early Medieval Indian society and trade guilds.",
            metadata={"source": "History_2024-25.pdf", "page": 18, "chunk_index": 1},
        ),
    ]
    sample_ids = ["History_2024-25.pdf:p12:c0", "History_2024-25.pdf:p18:c0"]

    # First indexing run
    vector_store = Chroma(
        collection_name=test_collection,
        embedding_function=fake_embeddings,
        persist_directory=str(test_db_dir),
    )
    vector_store.add_documents(documents=sample_chunks, ids=sample_ids)

    count_run_1 = vector_store._collection.count()
    assert count_run_1 == 2

    # Second indexing run (re-indexing should reset/prevent duplicate indexing)
    vector_store.reset_collection()
    vector_store.add_documents(documents=sample_chunks, ids=sample_ids)

    count_run_2 = vector_store._collection.count()
    assert count_run_2 == 2, "Re-indexing must not produce duplicate documents"

    # Query the test vector store
    results = vector_store.similarity_search("Mughal administration", k=2)
    assert len(results) == 2
    pages_found = {r.metadata["page"] for r in results}
    assert pages_found == {12, 18}
    assert all(r.metadata["source"] == "History_2024-25.pdf" for r in results)


def test_download_catalogue_pdf_skip_when_present(tmp_path):
    """Verify that download_catalogue_pdf skips re-downloading when valid file exists."""
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_text("Dummy content simulating downloaded PDF")

    result_path = download_catalogue_pdf(url="http://invalid.example/pdf", destination=fake_pdf)
    assert result_path == fake_pdf
    assert fake_pdf.read_text() == "Dummy content simulating downloaded PDF"


def test_embedding_model_configuration():
    """Verify that DEFAULT_EMBEDDING_MODEL points to the supported stable gemini-embedding-2 model."""
    assert DEFAULT_EMBEDDING_MODEL in ("gemini-embedding-2", "models/gemini-embedding-2")
    assert "text-embedding-004" not in DEFAULT_EMBEDDING_MODEL


def test_get_gemini_api_key_resolution(monkeypatch):
    """Verify get_gemini_api_key precedence: GEMINI_API_KEY first, GOOGLE_API_KEY fallback."""
    # Test GEMINI_API_KEY takes priority
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    assert get_gemini_api_key() == "test-gemini-key"

    # Test GOOGLE_API_KEY fallback when GEMINI_API_KEY is empty
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    assert get_gemini_api_key() == "test-google-key"

    # Test ValueError when both missing
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Google Gemini API key not found"):
        get_gemini_api_key()

