import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

try:
    from rag_assistant.embeddings import generate_embedding
except ImportError:
    from embeddings import generate_embedding

try:
    from langchain_core.documents import Document
except ImportError:
    Document = None

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("chroma_manager")

CHROMA_DB_PATH = Path("chroma_db")
COLLECTION_NAME = "ai_digest_knowledge"
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150


def _get_collection():
    """Return the persistent ChromaDB collection used by the RAG assistant."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _document_id(title: str, source: str, url: str) -> str:
    raw_id = f"{title or ''}|{source or ''}|{url or ''}".strip()
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


def _today_iso() -> str:
    return datetime.now().date().isoformat()


def _normalize_date(value: Optional[str]) -> str:
    """Return YYYY-MM-DD when a date-like string is available."""
    if not value:
        return ""

    value = str(value).strip()
    if not value:
        return ""

    match = re.match(r"^\d{4}-\d{2}-\d{2}", value)
    if match:
        return match.group(0)

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return value


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping chunks before embedding."""
    if not text or not str(text).strip():
        return []


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(str(text).strip()) if chunk.strip()]


def add_document(
    title: str,
    summary: str,
    source: str = "Unknown",
    url: str = "",
    published: str = "",
    ingested_date: str = None,
) -> bool:
    """Add a digest or knowledge-base document to ChromaDB as embedded chunks."""
    if not summary or not str(summary).strip():
        logger.warning("Skipping ChromaDB insert because summary is empty: %s", title)
        return False

    try:
        collection = _get_collection()
        doc_id = _document_id(title, source, url)
        first_chunk_id = f"{doc_id}:chunk:0"
        existing = collection.get(ids=[doc_id, first_chunk_id])
        if existing and existing.get("ids"):
            logger.info("Skipping duplicate ChromaDB document: %s", title)
            return False

        normalized_published = _normalize_date(published)
        normalized_ingested_date = _normalize_date(ingested_date) or _today_iso()
        document_text = (
            f"Title: {title or 'Untitled'}\n"
            f"Source: {source or 'Unknown'}\n"
            f"Published: {published or 'Unknown'}\n"
            f"Ingested Date: {normalized_ingested_date}\n\n"
            f"{summary}"
        )
        chunks = chunk_text(document_text)
        if not chunks:
            logger.warning("Skipping ChromaDB insert because chunking produced no text: %s", title)
            return False

        ids = [f"{doc_id}:chunk:{idx}" for idx, _ in enumerate(chunks)]
        embeddings = [generate_embedding(chunk) for chunk in chunks]
        metadatas = []
        for idx, chunk in enumerate(chunks):
            metadatas.append({
                "document_id": doc_id,
                "chunk_index": idx,
                "chunk_count": len(chunks),
                "title": title or "Untitled",
                "source": source or "Unknown",
                "url": url or "",
                "published": published or "",
                "published_date": normalized_published,
                "ingested_date": normalized_ingested_date,
                "summary": summary,
            })

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Stored document in ChromaDB as %d chunk(s): %s", len(chunks), title)
        return True
    except Exception as exc:
        logger.error("Failed to add document to ChromaDB: %s", exc)
        raise


def retrieve_documents(
    query: str,
    top_k: int = 3,
    source_filter: str = None,
    date_filter: str = None,
) -> List[Dict[str, Any]]:
    """Retrieve semantically similar knowledge documents with progressive fallback."""
    if not query or not isinstance(query, str) or not query.strip():
        return []

    try:
        collection = _get_collection()
        query_embedding = generate_embedding(query)

        # We fetch more candidates than top_k when post-filtering is needed
        fetch_k = top_k * 5 if (source_filter or date_filter) else top_k

        if source_filter:
            logger.info("Applying source filter: %s", source_filter)
        if date_filter:
            logger.info("Applying date filter: %s", date_filter)

        # --- TRY 1: source + date (Python post-filter) ---
        matches = _query_and_filter(
            collection, query_embedding, fetch_k,
            where_clause=None,
            date_filter=date_filter,
            source_filter=source_filter,
        )

        if matches:
            return matches[:top_k]

        # --- TRY 2: source-only (drop date constraint) ---
        if date_filter and source_filter:
            logger.info("Strict retrieval returned 0 results")
            logger.info("Falling back to source-only retrieval")
            
            matches = _query_and_filter(
                collection, query_embedding, fetch_k,
                where_clause=None,
                date_filter=None,
                source_filter=source_filter,
            )
            if matches:
                return matches[:top_k]

        # --- TRY 3: pure semantic (no metadata constraints) ---
        if source_filter or date_filter:
            if not date_filter or not source_filter:
                # If we didn't just log the 0 results above
                logger.info("Strict retrieval returned 0 results")
            logger.info("Falling back to semantic retrieval")

        matches = _query_and_filter(
            collection, query_embedding, top_k,
            where_clause=None,
            date_filter=None,
            source_filter=None,
        )
        return matches[:top_k]

    except Exception as exc:
        logger.error("Failed to retrieve documents from ChromaDB: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Internal retrieval helpers
# ---------------------------------------------------------------------------



def _query_and_filter(
    collection,
    query_embedding: List[float],
    n_results: int,
    where_clause: dict = None,
    date_filter: str = None,
    source_filter: str = None,
) -> List[Dict[str, Any]]:
    """Run a ChromaDB semantic query and apply lightweight Python-side post-filtering."""
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }

    if where_clause:
        query_kwargs["where"] = where_clause

    try:
        results = collection.query(**query_kwargs)
    except Exception:
        query_kwargs.pop("where", None)
        results = collection.query(**query_kwargs)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches: List[Dict[str, Any]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        # --- Python-side source matching (case-insensitive) ---
        if source_filter:
            stored_source = (metadata.get("source") or "").strip()
            if stored_source.lower() != source_filter.lower():
                continue

        # --- Python-side date matching (ingested_date first, published fallback) ---
        # Older ChromaDB rows may not have ingested_date, so published remains
        # a backward-compatible fallback for already stored documents.
        if date_filter:
            stored_ingested_date = (metadata.get("ingested_date") or "").strip()
            stored_published_date = (metadata.get("published_date") or "").strip()
            stored_published = (metadata.get("published") or "").strip()
            candidate_dates = [
                stored_ingested_date,
                stored_published_date,
                stored_published,
            ]
            if not any(candidate.startswith(date_filter) for candidate in candidate_dates if candidate):
                continue
            logger.info("Matched document date filter %s for %s", date_filter, metadata.get("title", "Untitled"))

        langchain_document = None
        if Document:
            langchain_document = Document(
                page_content=document, metadata=metadata
            )

        matches.append({
            "summary": document,
            "title": metadata.get("title", "Untitled"),
            "metadata": metadata,
            "similarity": 1 / (1 + distance) if distance is not None else None,
            "document": langchain_document,
        })

    return matches
