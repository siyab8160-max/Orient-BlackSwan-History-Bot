"""
Orient BlackSwan History Catalogue RAG Bot - FastAPI Application
================================================================
Main entry point for the FastAPI backend and static frontend server.

Provides:
- Health check endpoint: GET /api/health
- RAG Query endpoint scaffold: POST /api/query
- Static file serving for the vanilla HTML/CSS/JS frontend
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

from app.rag import QueryRequest, QueryResponse, answer_query

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
PDF_LOCAL_PATH = Path(os.getenv("PDF_LOCAL_PATH", str(DATA_DIR / "History_2024-25.pdf")))
CHROMA_PERSIST_DIRECTORY = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", str(DATA_DIR / "chroma_db")))

# FastAPI App Initialization
app = FastAPI(
    title="Orient BlackSwan History Catalogue RAG Bot",
    description="Academic RAG system for querying the Orient BlackSwan History 2024-25 Catalogue with strict citations.",
    version="0.1.0",
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", tags=["System"])
def health_check():
    """
    Check the health status of the API server and inspect the environment configuration.

    Returns:
        JSON object with status, version, and readiness flags for PDF, Vector Store, and Gemini API key.
    """
    has_gemini_key = bool(os.getenv("GEMINI_API_KEY", "").strip())
    pdf_exists = PDF_LOCAL_PATH.exists()
    chroma_exists = CHROMA_PERSIST_DIRECTORY.exists() and any(CHROMA_PERSIST_DIRECTORY.iterdir()) if CHROMA_PERSIST_DIRECTORY.exists() else False

    return {
        "status": "ok",
        "app": "Orient BlackSwan History Catalogue RAG Bot",
        "version": "0.1.0",
        "checks": {
            "gemini_api_key_configured": has_gemini_key,
            "catalogue_pdf_present": pdf_exists,
            "vector_database_indexed": chroma_exists,
        },
        "pdf_source": os.getenv("PDF_URL", "https://www.orientblackswan.com/pdfs/downloads/History_2024-25.pdf"),
        "message": "Orient BlackSwan History Bot scaffold is operational.",
    }


@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
def query_catalogue(request: QueryRequest):
    """
    Scaffold endpoint for answering questions against the catalogue.
    Validates the query and returns the structured answer schema with citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    return answer_query(request.question)


# Serve root index.html explicitly
@app.get("/", include_in_schema=False)
def serve_root():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


# Mount frontend static directory to serve CSS, JS, and any static assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend-root")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() in ("true", "1")

    print(f"[STARTUP] Running Orient BlackSwan History Bot on http://{host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=debug)
