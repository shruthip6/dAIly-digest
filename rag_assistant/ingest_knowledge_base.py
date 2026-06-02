import logging
from pathlib import Path

try:
    from rag_assistant.chroma_manager import add_document
except ImportError:
    from chroma_manager import add_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("knowledge_ingestion")

KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"


def ingest_knowledge_base() -> int:
    """Load all foundational AI knowledge files into ChromaDB."""
    if not KNOWLEDGE_BASE_DIR.exists():
        logger.error("Knowledge base directory not found: %s", KNOWLEDGE_BASE_DIR)
        return 0

    ingested_count = 0
    for file_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                logger.warning("Skipping empty knowledge file: %s", file_path.name)
                continue

            stored = add_document(
                title=file_path.stem.replace("_", " ").title(),
                summary=content,
                source="Foundational AI Knowledge Base",
                url="",
                published=""
            )
            if stored:
                ingested_count += 1
                print(f"Ingested: {file_path.name}")
            else:
                print(f"Skipped duplicate: {file_path.name}")
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", file_path.name, exc)
            print(f"Failed: {file_path.name} - {exc}")

    print(f"Knowledge base ingestion complete. New documents stored: {ingested_count}")
    return ingested_count


if __name__ == "__main__":
    ingest_knowledge_base()
