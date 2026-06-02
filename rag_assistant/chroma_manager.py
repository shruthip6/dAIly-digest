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


def retrieve_documents(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve semantically similar knowledge documents for a user query."""
    if not query or not isinstance(query, str) or not query.strip():
        return []

    try:
        collection = _get_collection()
        query_embedding = generate_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            langchain_document = None
            if Document:
                # LangChain is used only as a lightweight retrieval abstraction wrapper;
                # ChromaDB remains the persistent vector store and retrieval engine.
                langchain_document = Document(page_content=document, metadata=metadata)

            matches.append({
                "summary": document,
                "title": metadata.get("title", "Untitled"),
                "metadata": metadata,
                "similarity": 1 / (1 + distance) if distance is not None else None,
                "document": langchain_document
            })

        return matches
    except Exception as exc:
        logger.error("Failed to retrieve documents from ChromaDB: %s", exc)
        return []
