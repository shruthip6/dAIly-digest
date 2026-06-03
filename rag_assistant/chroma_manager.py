import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List

import chromadb

try:
    from rag_assistant.embeddings import generate_embedding
except ImportError:
    from embeddings import generate_embedding

try:
    from langchain_core.documents import Document
except ImportError:
    Document = None

logger = logging.getLogger("chroma_manager")

CHROMA_DB_PATH = Path("chroma_db")
COLLECTION_NAME = "ai_digest_knowledge"


def _get_collection():
    """Return the persistent ChromaDB collection used by the RAG assistant."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _document_id(title: str, source: str, url: str) -> str:
    raw_id = f"{title or ''}|{source or ''}|{url or ''}".strip()
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


def add_document(
    title: str,
    summary: str,
    source: str = "Unknown",
    url: str = "",
    published: str = ""
) -> bool:
    """Add a digest or knowledge-base document to ChromaDB, skipping duplicates safely."""
    if not summary or not str(summary).strip():
        logger.warning("Skipping ChromaDB insert because summary is empty: %s", title)
        return False

    try:
        collection = _get_collection()
        doc_id = _document_id(title, source, url)
        existing = collection.get(ids=[doc_id])
        if existing and existing.get("ids"):
            logger.info("Skipping duplicate ChromaDB document: %s", title)
            return False

        document_text = (
            f"Title: {title or 'Untitled'}\n"
            f"Source: {source or 'Unknown'}\n"
            f"Published: {published or 'Unknown'}\n\n"
            f"{summary}"
        )
        embedding = generate_embedding(document_text)

        collection.add(
            ids=[doc_id],
            documents=[document_text],
            embeddings=[embedding],
            metadatas=[{
                "title": title or "Untitled",
                "source": source or "Unknown",
                "url": url or "",
                "published": published or "",
                "summary": summary
            }]
        )
        logger.info("Stored document in ChromaDB: %s", title)
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

        # --- Python-side date matching (startswith for timestamp compat) ---
        # Future ingestions should normalize published dates into: YYYY-MM-DD
        # but the retrieval layer must remain backward compatible
        # with older timestamp metadata already stored in ChromaDB.
        if date_filter:
            stored_published = (metadata.get("published") or "").strip()
            if not stored_published.startswith(date_filter):
                continue
            logger.info("Matched document published date:\n%s", stored_published)

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

