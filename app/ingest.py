"""
Orient BlackSwan History Catalogue - Document Ingestion Pipeline
================================================================
This module downloads, extracts, splits, and indexes the Orient BlackSwan
History 2024-25 catalogue into a persistent local ChromaDB vector database.

Tech Stack:
- PyPDFLoader (langchain_community.document_loaders)
- RecursiveCharacterTextSplitter (langchain_text_splitters)
- GoogleGenerativeAIEmbeddings (langchain_google_genai)
- Chroma (langchain_chroma)

Academic Grounding Constraints:
- Preserves the source filename and zero-based page metadata for every chunk.
- Avoids duplicate indexing via safe collection recreation and stable IDs.
- Reports total page count, chunk count, and persistent ChromaDB storage location.
"""

import os
import sys
import time
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
from dotenv import load_dotenv

# Document loading, splitting, embeddings, and vector store
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment configuration from .env file
load_dotenv()

# Configuration constants with environment variable overrides
PDF_URL: str = os.getenv(
    "PDF_URL",
    "https://www.orientblackswan.com/pdfs/downloads/History_2024-25.pdf",
)
PDF_LOCAL_PATH: Path = Path(os.getenv("PDF_LOCAL_PATH", "data/History_2024-25.pdf"))
CHROMA_PERSIST_DIRECTORY: Path = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma_db"))
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "obs_history_catalogue")

# Embedding model verified against official Google Gemini & LangChain documentation:
# Supported stable model: "gemini-embedding-2" (replaces retired "text-embedding-004")
DEFAULT_EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")

# Chunking parameters
DEFAULT_CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))


def get_gemini_api_key() -> str:
    """
    Retrieve the Google Gemini API key from environment variables.
    Checks GEMINI_API_KEY first, followed by GOOGLE_API_KEY.

    Returns:
        The API key string.

    Raises:
        ValueError: If no valid API key is found in the environment.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "Google Gemini API key not found or still set to placeholder.\n"
            "Please set GEMINI_API_KEY in your .env file or environment variables.\n"
            "You can obtain a key at: https://aistudio.google.com/"
        )
    return api_key


def download_catalogue_pdf(
    url: str = PDF_URL,
    destination: Path = PDF_LOCAL_PATH,
    timeout: Tuple[int, int] = (10, 120),
) -> Path:
    """
    Download the official Orient BlackSwan History catalogue PDF if not already present.

    Args:
        url: Direct download URL for the catalogue PDF.
        destination: Local path where the PDF will be saved.
        timeout: Tuple of (connection_timeout_seconds, read_timeout_seconds).

    Returns:
        Path to the downloaded PDF file.

    Raises:
        requests.RequestException: If the HTTP request fails or times out.
        IOError: If writing to the destination fails.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        file_size_mb = destination.stat().st_size / (1024 * 1024)
        print(f"[INFO] Catalogue PDF already present at: {destination} ({file_size_mb:.2f} MB). Skipping download.")
        return destination

    print(f"[INFO] Downloading official catalogue PDF from:\n       {url}")
    print(f"[INFO] Destination: {destination}")

    # Use a temporary file to avoid keeping a corrupt/partial file on failure
    temp_destination = destination.with_suffix(".tmp")
    headers = {"User-Agent": "Mozilla/5.0 (OrientBlackSwan-RAG-Bot/0.1)"}

    try:
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(temp_destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r[INFO] Downloading: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)", end="", flush=True)

        print()  # newline after progress
        temp_destination.replace(destination)
        total_mb = destination.stat().st_size / (1024 * 1024)
        print(f"[SUCCESS] Catalogue PDF downloaded successfully ({total_mb:.2f} MB).")
        return destination

    except requests.Timeout as e:
        if temp_destination.exists():
            temp_destination.unlink()
        print(f"\n[ERROR] Request timed out while downloading PDF: {e}", file=sys.stderr)
        raise
    except requests.RequestException as e:
        if temp_destination.exists():
            temp_destination.unlink()
        print(f"\n[ERROR] Network error while downloading PDF: {e}", file=sys.stderr)
        raise
    except Exception as e:
        if temp_destination.exists():
            temp_destination.unlink()
        print(f"\n[ERROR] Unexpected error saving PDF: {e}", file=sys.stderr)
        raise


def extract_pages(pdf_path: Path = PDF_LOCAL_PATH) -> List[Document]:
    """
    Extract text and metadata from the PDF using PyPDFLoader.
    Preserves source filename and zero-based page metadata on each Document.

    Args:
        pdf_path: Path to the local PDF file.

    Returns:
        List of Document objects, one for each page, containing page_content and metadata.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}. Please download it first.")

    print(f"[INFO] Loading PDF pages via PyPDFLoader from: {pdf_path.name}...")
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    # Ensure source is standard filename and page is zero-based integer
    source_filename = pdf_path.name
    for idx, doc in enumerate(pages):
        doc.metadata["source"] = source_filename
        # PyPDFLoader defaults to zero-based 'page' index:
        if "page" not in doc.metadata:
            doc.metadata["page"] = idx
        else:
            doc.metadata["page"] = int(doc.metadata["page"])

    print(f"[SUCCESS] Extracted {len(pages)} pages from {source_filename}.")
    return pages


def split_into_overlapping_chunks(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Tuple[List[Document], List[str]]:
    """
    Split pages into overlapping chunks using RecursiveCharacterTextSplitter.
    Preserves original page metadata and assigns stable deterministic IDs.

    Args:
        documents: List of page Document objects.
        chunk_size: Target character length per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        Tuple of (chunks_list, stable_ids_list).
    """
    # Filter out empty pages (e.g. image-only covers) so chunks always contain text
    non_empty_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    print(f"[INFO] Splitting {len(non_empty_documents)} text-bearing pages into chunks (size={chunk_size}, overlap={chunk_overlap})...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(non_empty_documents)

    # Generate stable deterministic chunk IDs to avoid duplicates on repeat runs
    # Format: <filename>:p<zero_based_page>:c<chunk_idx>
    stable_ids: List[str] = []
    page_chunk_counters: Dict[int, int] = {}

    for idx, chunk in enumerate(chunks):
        page_num = chunk.metadata.get("page", 0)
        source_name = chunk.metadata.get("source", "catalogue.pdf")

        chunk_in_page = page_chunk_counters.get(page_num, 0)
        page_chunk_counters[page_num] = chunk_in_page + 1

        chunk_id = f"{source_name}:p{page_num}:c{chunk_in_page}"
        stable_ids.append(chunk_id)

        # Store chunk sequence index in metadata
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_in_page"] = chunk_in_page

    print(f"[SUCCESS] Created {len(chunks)} text chunks across {len(documents)} pages.")
    return chunks, stable_ids


def create_or_update_vectorstore(
    chunks: List[Document],
    ids: List[str],
    persist_directory: Path = CHROMA_PERSIST_DIRECTORY,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reset_collection: bool = True,
    api_key: Optional[str] = None,
) -> Chroma:
    """
    Generate embeddings using GoogleGenerativeAIEmbeddings and store chunks in persistent ChromaDB.
    Avoids duplicate indexing by safely resetting the collection or using stable IDs.

    Args:
        chunks: List of chunk Document objects.
        ids: Stable deterministic IDs for each chunk.
        persist_directory: Local directory for ChromaDB storage.
        collection_name: Name of the Chroma collection.
        embedding_model: Google GenAI embedding model identifier.
        reset_collection: Whether to safely reset existing collection before indexing.
        api_key: Optional Google Gemini API key (defaults to environment).

    Returns:
        Chroma vector store instance.
    """
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    if not api_key:
        api_key = get_gemini_api_key()

    print(f"[INFO] Initializing LangChain Google embeddings (model='{embedding_model}')...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=api_key,
    )

    print(f"[INFO] Connecting to persistent ChromaDB at: {persist_directory}")
    print(f"[INFO] Target collection: '{collection_name}'")

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    if reset_collection:
        print(f"[INFO] Resetting collection '{collection_name}' to avoid duplicate indexing...")
        try:
            vector_store.reset_collection()
            print(f"[INFO] Collection '{collection_name}' reset successfully.")
        except Exception as e:
            print(f"[WARNING] Note during collection reset: {e}. Reinitializing collection.")
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(persist_directory),
            )

    # Fetch existing document IDs to prevent re-embedding already indexed chunks
    existing_ids = set()
    try:
        res = vector_store.get()
        if res and "ids" in res:
            existing_ids = set(res["ids"])
    except Exception as e:
        print(f"[DEBUG] Could not fetch existing IDs: {e}")

    unindexed_chunks: List[Document] = []
    unindexed_ids: List[str] = []
    for chunk, cid in zip(chunks, ids):
        if cid not in existing_ids:
            unindexed_chunks.append(chunk)
            unindexed_ids.append(cid)

    if not unindexed_chunks:
        print(f"[INFO] All {len(chunks)} chunks are already present in ChromaDB. Skipping re-embedding.")
        return vector_store

    print(f"[INFO] Embedding and adding {len(unindexed_chunks)} chunks ({len(existing_ids)} already indexed) into ChromaDB...")
    batch_size = 80
    total_unindexed = len(unindexed_chunks)

    for i in range(0, total_unindexed, batch_size):
        batch_chunks = unindexed_chunks[i : i + batch_size]
        batch_ids = unindexed_ids[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_unindexed + batch_size - 1) // batch_size

        max_retries = 6
        for attempt in range(max_retries):
            try:
                print(f"[INFO] Indexing batch {batch_num} of {total_batches} ({len(batch_chunks)} items)...")
                vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
                break
            except Exception as exc:
                err_str = str(exc)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()) and attempt < max_retries - 1:
                    match = re.search(r"retry in (\d+(\.\d+)?)s", err_str, re.IGNORECASE)
                    wait_seconds = float(match.group(1)) + 5 if match else 45.0
                    print(f"[RATE LIMIT] Gemini quota reached on batch {batch_num}. Waiting {wait_seconds:.1f}s before retrying (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_seconds)
                else:
                    raise exc

        if i + batch_size < total_unindexed:
            time.sleep(2)

    total_final = vector_store._collection.count()
    print(f"[SUCCESS] All chunks successfully stored in ChromaDB (total records in collection: {total_final}).")
    return vector_store


def run_ingestion(
    pdf_url: str = PDF_URL,
    pdf_path: Path = PDF_LOCAL_PATH,
    persist_dir: Path = CHROMA_PERSIST_DIRECTORY,
    collection_name: str = CHROMA_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    reset_collection: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute the end-to-end ingestion pipeline:
    1. Download PDF if not present.
    2. Extract pages and preserve source filename and zero-based page metadata.
    3. Split pages into overlapping chunks with stable IDs.
    4. Generate embeddings and store in persistent ChromaDB.
    5. Report page count, chunk count, and ChromaDB location.

    Returns:
        Summary dict containing page count, chunk count, ChromaDB path, and collection name.
    """
    print("\n" + "=" * 70)
    print(" Orient BlackSwan History Catalogue - Document Ingestion Pipeline")
    print("=" * 70)

    # Step 1: Download PDF
    downloaded_pdf = download_catalogue_pdf(url=pdf_url, destination=pdf_path)

    # Step 2: Extract PDF pages
    pages = extract_pages(pdf_path=downloaded_pdf)
    page_count = len(pages)

    # Step 3: Split into chunks
    chunks, stable_ids = split_into_overlapping_chunks(
        documents=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunk_count = len(chunks)

    # Step 4: Index into ChromaDB (unless dry_run requested)
    if dry_run:
        print("\n[NOTICE] Dry-run mode enabled: Skipping embedding generation and ChromaDB indexing.")
    else:
        create_or_update_vectorstore(
            chunks=chunks,
            ids=stable_ids,
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            reset_collection=reset_collection,
        )

    # Step 5: Report Summary
    abs_chroma_path = str(Path(persist_dir).resolve())
    print("\n" + "=" * 70)
    print(" INGESTION PIPELINE REPORT")
    print("=" * 70)
    print(f" Source Document:      {downloaded_pdf.name}")
    print(f" Total Pages:          {page_count}")
    print(f" Total Chunks:         {chunk_count}")
    print(f" Embedding Model:      {embedding_model}")
    print(f" Collection Name:      {collection_name}")
    print(f" ChromaDB Location:    {abs_chroma_path}")
    print("=" * 70 + "\n")

    return {
        "page_count": page_count,
        "chunk_count": chunk_count,
        "chroma_dir": abs_chroma_path,
        "collection_name": collection_name,
        "embedding_model": embedding_model,
    }


def main():
    """Command-line entry point for independent ingestion execution."""
    parser = argparse.ArgumentParser(
        description="Ingest the Orient BlackSwan History Catalogue PDF into ChromaDB."
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not reset the ChromaDB collection before indexing (uses stable ID upsert).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download, extract, and chunk without calling Gemini embeddings or indexing.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Target character length per chunk (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Character overlap between consecutive chunks (default: {DEFAULT_CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--pdf-url",
        type=str,
        default=PDF_URL,
        help="Custom URL for catalogue PDF download.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=str(CHROMA_PERSIST_DIRECTORY),
        help="Custom local directory for ChromaDB storage.",
    )

    args = parser.parse_args()

    try:
        run_ingestion(
            pdf_url=args.pdf_url,
            pdf_path=PDF_LOCAL_PATH,
            persist_dir=Path(args.persist_dir),
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            reset_collection=not args.no_reset,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Ingestion interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL] Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
