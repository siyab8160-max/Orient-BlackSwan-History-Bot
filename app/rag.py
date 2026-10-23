"""
Orient BlackSwan History Catalogue - Academic RAG Query Engine
=============================================================
This module implements the complete retrieval-augmented generation pipeline
strictly grounded on the Orient BlackSwan History 2024-25 catalogue PDF.

Core Academic Rules:
1. Strict Grounding: Answers are derived EXCLUSIVELY from retrieved PDF excerpts.
2. Physical Page Citations: Citations use 1-indexed physical PDF page numbers.
3. Verbatim Excerpt: Citations include a direct supporting excerpt from that page.
4. Exact Fallback: Returns EXACTLY "Not found" with found=False and citations=[]
   if the catalogue does not contain direct evidence.
5. Zero External Knowledge: General world knowledge outside the PDF is suppressed.
6. Citation Validation: Every citation is verified against actual retrieved chunk text.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from app.ingest import (
    get_gemini_api_key,
    DEFAULT_EMBEDDING_MODEL,
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_COLLECTION_NAME,
)

load_dotenv()

# Constant fallback string required when information is absent from the catalogue
NOT_FOUND_RESPONSE: str = "Not found"
DEFAULT_LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

# Global singleton cache for vector store and LLM to optimize query latency
_VECTOR_STORE: Optional[Chroma] = None
_LLM_CLIENTS: Dict[str, ChatGoogleGenerativeAI] = {}


class Citation(BaseModel):
    """Citation metadata for an answer."""
    page: int = Field(..., description="Physical 1-indexed page number from the PDF catalogue")
    excerpt: str = Field(..., description="Direct verbatim supporting excerpt from that page")


class QueryRequest(BaseModel):
    """Incoming user query."""
    question: str = Field(..., min_length=2, max_length=500, description="User's query about the history catalogue")


class QueryResponse(BaseModel):
    """Structured response containing answer, citations, and search status."""
    question: str
    answer: str = Field(..., description="The grounded answer or 'Not found'")
    citations: List[Citation] = Field(default_factory=list, description="List of physical page citations and excerpts")
    found: bool = Field(default=True, description="False if document did not contain evidence ('Not found')")
    model: str = Field(default=DEFAULT_LLM_MODEL, description="Underlying model identifier")


def get_vectorstore() -> Chroma:
    """
    Load and return the persistent ChromaDB vector store.
    Uses singleton caching for query efficiency.
    """
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    api_key = get_gemini_api_key()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=DEFAULT_EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    persist_dir = str(CHROMA_PERSIST_DIRECTORY)
    _VECTOR_STORE = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return _VECTOR_STORE


def get_llm(model_name: str = DEFAULT_LLM_MODEL) -> ChatGoogleGenerativeAI:
    """
    Initialize and return the Google Gemini generative chat model.
    Configured with max_retries=1 and timeout=12s for fast responsive queries.
    """
    global _LLM_CLIENTS
    if model_name in _LLM_CLIENTS:
        return _LLM_CLIENTS[model_name]

    api_key = get_gemini_api_key()
    client = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        max_retries=2,
        timeout=45.0,
    )
    _LLM_CLIENTS[model_name] = client
    return client


def retrieve_context(question: str, k: int = 8) -> List[Document]:
    """
    Retrieve the top-k most relevant document chunks from ChromaDB for a query.
    Ensures comprehensive context coverage across the catalogue.
    """
    vector_store = get_vectorstore()
    return vector_store.similarity_search(question, k=k)


def clean_llm_json_text(raw_content: Any) -> str:
    """
    Extract and clean raw JSON text from LangChain message response content,
    stripping markdown formatting or code fence indicators.
    """
    if isinstance(raw_content, list):
        parts = []
        for item in raw_content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        text = "".join(parts)
    else:
        text = str(raw_content)

    text = text.strip()
    # Strip markdown code blocks (```json ... ``` or ``` ... ```)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_citations_against_chunks(
    citations: List[Dict[str, Any]],
    retrieved_docs: List[Document],
) -> List[Citation]:
    """
    Strictly validate citation page numbers and verbatim excerpts against retrieved chunks.
    Ensures citations reference physical 1-indexed pages and exact text from the catalogue.
    """
    validated: List[Citation] = []
    
    # Map physical page -> concatenated text on that page among retrieved docs
    page_texts: Dict[int, str] = {}
    for doc in retrieved_docs:
        phys_page = int(doc.metadata.get("page", 0)) + 1
        page_texts[phys_page] = page_texts.get(phys_page, "") + "\n" + doc.page_content

    for cit in citations:
        try:
            page = int(cit.get("page", 0))
            excerpt = str(cit.get("excerpt", "")).strip()
        except (ValueError, TypeError):
            continue

        if not excerpt or page <= 0:
            continue

        # Check if physical page is in the retrieved docs
        if page in page_texts:
            full_page_text = page_texts[page]
            # Normalize whitespace for flexible matching
            norm_page_text = " ".join(full_page_text.split())
            norm_excerpt = " ".join(excerpt.split())
            
            if norm_excerpt.lower() in norm_page_text.lower():
                validated.append(Citation(page=page, excerpt=excerpt))
            else:
                # Substring fragment match if excerpt was partially truncated
                words = norm_excerpt.split()
                if len(words) >= 4:
                    prefix = " ".join(words[:4])
                    if prefix.lower() in norm_page_text.lower():
                        validated.append(Citation(page=page, excerpt=excerpt))
        else:
            # Check if excerpt exists on any other retrieved page
            for other_page, other_text in page_texts.items():
                norm_other = " ".join(other_text.split())
                norm_excerpt = " ".join(excerpt.split())
                if norm_excerpt.lower() in norm_other.lower():
                    validated.append(Citation(page=other_page, excerpt=excerpt))
                    break

    # Deduplicate citations by (page, excerpt)
    unique_citations: List[Citation] = []
    seen = set()
    for c in validated:
        key = (c.page, c.excerpt)
        if key not in seen:
            seen.add(key)
            unique_citations.append(c)

    return unique_citations


def answer_query(
    question: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    k: int = 8,
) -> QueryResponse:
    """
    Execute the complete RAG pipeline:
    1. Retrieve relevant chunks from persistent ChromaDB.
    2. Format prompt with physical 1-based page numbers.
    3. Invoke Gemini LLM for strict grounded extraction.
    4. Validate citations and strictly enforce 'Not found' fallback.

    Args:
        question: User query string.
        llm_model: Google Gemini model identifier.
        k: Number of retrieved chunks to provide as context.

    Returns:
        Structured QueryResponse with answer, physical page citations, and found flag.
    """
    clean_question = question.strip()
    if not clean_question:
        return QueryResponse(
            question=question,
            answer=NOT_FOUND_RESPONSE,
            citations=[],
            found=False,
            model=llm_model,
        )

    # 1. Retrieve document chunks
    try:
        retrieved_docs = retrieve_context(clean_question, k=k)
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        return QueryResponse(
            question=clean_question,
            answer=NOT_FOUND_RESPONSE,
            citations=[],
            found=False,
            model=llm_model,
        )

    if not retrieved_docs:
        return QueryResponse(
            question=clean_question,
            answer=NOT_FOUND_RESPONSE,
            citations=[],
            found=False,
            model=llm_model,
        )

    # 2. Build context blocks with 1-based physical page numbers
    context_blocks = []
    for idx, doc in enumerate(retrieved_docs):
        # Convert 0-based page metadata from PyPDFLoader to physical 1-based page number
        zero_based_page = doc.metadata.get("page", 0)
        physical_page = int(zero_based_page) + 1
        source_name = doc.metadata.get("source", "History_2024-25.pdf")
        context_blocks.append(
            f"--- [Passage {idx + 1} | Source: {source_name} | Physical Page: {physical_page}] ---\n"
            f"{doc.page_content.strip()}"
        )

    context_str = "\n\n".join(context_blocks)

    # 3. Construct the strict academic RAG prompt
    prompt = (
        "You are an academic bibliographic research assistant for the Orient BlackSwan History 2024-25 Catalogue.\n"
        "Your task is to answer the user's question using EXCLUSIVELY the provided context passages from the catalogue.\n\n"
        "CRITICAL ACADEMIC RULES:\n"
        "1. STRICT GROUNDING: Answer ONLY using facts directly stated in the context passages. "
        "Do NOT speculate, assume, or use any external knowledge.\n"
        "2. NOT FOUND CRITERIA: If the context passages do not contain explicit, direct evidence to answer the question, "
        "or if the question is subjective (such as asking for the 'best' book where no specific book is uniquely designated as 'the best'), "
        "or if the topic is outside this history catalogue (such as general science textbooks or current political officeholders), "
        "you MUST respond with EXACTLY:\n"
        "   {\n"
        '     "answer": "Not found",\n'
        '     "citations": [],\n'
        '     "found": false\n'
        "   }\n"
        "   Do NOT explain why, do NOT apologize, and do NOT provide outside information.\n"
        "3. GROUNDED CITATIONS: If and only if the context contains direct evidence, provide:\n"
        '   - "answer": A concise, accurate factual sentence answering the query.\n'
        '   - "citations": A list of citations where "page" is the physical 1-based page number indicated in the passage header, '
        'and "excerpt" is the exact verbatim sentence from that passage that proves the claim.\n'
        '   - "found": true\n'
        "4. OUTPUT FORMAT: Output ONLY a valid JSON object matching the schema:\n"
        "{\n"
        '  "answer": "...",\n'
        '  "citations": [\n'
        '    {"page": 2, "excerpt": "..."}\n'
        "  ],\n"
        '  "found": true\n'
        "}\n\n"
        f"CATALOGUE CONTEXT PASSAGES:\n{context_str}\n\n"
        f"USER QUESTION:\n{clean_question}\n"
    )

    # 4. Invoke LLM with fast fallback across available models
    models_to_try = [llm_model]
    for alt in ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.1-flash-lite"]:
        if alt not in models_to_try:
            models_to_try.append(alt)

    data = None
    active_model = llm_model
    last_error = None

    for candidate_model in models_to_try:
        try:
            llm = get_llm(model_name=candidate_model)
            response = llm.invoke(prompt)
            cleaned_json = clean_llm_json_text(response.content)
            data = json.loads(cleaned_json)
            active_model = candidate_model
            break
        except Exception as e:
            last_error = e
            continue

    if data is None:
        print(f"[WARNING] All LLM Generation attempts failed ({last_error}). Falling back to 'Not found'.")
        return QueryResponse(
            question=clean_question,
            answer=NOT_FOUND_RESPONSE,
            citations=[],
            found=False,
            model=llm_model,
        )

    raw_answer = str(data.get("answer", "")).strip()
    is_found = bool(data.get("found", True))
    raw_citations = data.get("citations", [])

    # 5. Handle "Not found" explicitly
    if not raw_answer or raw_answer.lower() == "not found" or not is_found:
        return QueryResponse(
            question=clean_question,
            answer=NOT_FOUND_RESPONSE,
            citations=[],
            found=False,
            model=active_model,
        )

    # 6. Validate citations against retrieved evidence
    validated_citations = validate_citations_against_chunks(raw_citations, retrieved_docs)

    # If the LLM asserted a factual answer but produced no verifiable citations from the text,
    # strictly enforce academic fallback to "Not found"
    if not validated_citations:
        print(f"[INFO] No verifiable citations found for '{clean_question}'. Falling back to 'Not found'.")
        return QueryResponse(
            question=clean_question,
            answer=NOT_FOUND_RESPONSE,
            citations=[],
            found=False,
            model=active_model,
        )

    return QueryResponse(
        question=clean_question,
        answer=raw_answer,
        citations=validated_citations,
        found=True,
        model=active_model,
    )


# Backwards compatibility alias for app/main.py
answer_query_scaffold = answer_query
