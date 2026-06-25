import logging
from datetime import datetime
from typing import List, Dict, Any

# Local imports
from rss_fetcher import fetch_latest_ai_news
from utils.scraper import extract_article
from utils.summarizer import summarize_article

try:
    from rag_assistant.chroma_manager import add_document
except ImportError:
    from chroma_manager import add_document

# Configure module logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daily_digest")


def generate_daily_digest(max_articles: int = 3) -> List[Dict[str, Any]]:
    """Orchestrates the creation of a daily AI news digest.

    The pipeline performs the following steps:
    1. Fetch latest AI news metadata via ``fetch_latest_ai_news``.
    2. For each entry, scrape the article body using ``extract_article``.
    3. Summarize the scraped content with ``summarize_article``.
    4. Assemble a standardized digest record.

    Args:
        max_articles: Maximum number of articles to include in the digest.

    Returns:
        A list of dictionaries, each containing the original metadata, summary,
        provider used for summarisation, success flag and any error information.
    """
    logger.info("Starting daily digest generation ‑ max_articles=%s", max_articles)

    # Step 1 – fetch RSS feed items
    try:
        fetched_items = fetch_latest_ai_news(max_articles=max_articles)
        logger.info("Fetched %d items from RSS feeds", len(fetched_items))
    except Exception as exc:
        logger.error("Failed to fetch RSS feeds: %s", exc)
        return []

    digest: List[Dict[str, Any]] = []

    for item in fetched_items:
        article_record: Dict[str, Any] = {
            "title": item.get("title"),
            "source": item.get("source"),
            "url": item.get("url"),
            "published": item.get("published"),
            "summary": None,
            "provider": None,
            "success": False,
            "error": None,
            "ingested": False,
            "ingested_date": None,
        }

        url = article_record["url"]
        if not url:
            article_record["error"] = "Missing URL in feed item"
            logger.warning("Skipping item with missing URL: %s", article_record)
            digest.append(article_record)
            continue

        # Step 2 – scrape article content
        try:
            scrape_result = extract_article(url)
        except Exception as exc:
            scrape_result = {"success": False, "error": str(exc)}

        if not scrape_result.get("success"):
            article_record["error"] = f"Scraping failed: {scrape_result.get('error')}"
            logger.warning("Scraping failed for %s – %s", url, article_record["error"])
            digest.append(article_record)
            continue

        article_text = scrape_result.get("text")
        if not article_text:
            article_record["error"] = "Scraped article contains no text"
            logger.warning("Empty article text for %s", url)
            digest.append(article_record)
            continue

        # Step 3 – summarise article
        try:
            summary_result = summarize_article(article_text)
        except Exception as exc:
            summary_result = {"success": False, "error": str(exc)}

        if not summary_result.get("success"):
            article_record["error"] = f"Summarisation failed: {summary_result.get('error')}"
            logger.warning(
                "Summarisation failed for %s – %s", url, article_record["error"]
            )
            digest.append(article_record)
            continue

        # Populate successful record
        article_record["summary"] = summary_result.get("summary")
        article_record["provider"] = summary_result.get("provider")
        article_record["success"] = True
        article_record["error"] = None

        ingested_date = datetime.now().date().isoformat()
        article_record["ingested_date"] = ingested_date
        try:
            article_record["ingested"] = add_document(
                title=article_record.get("title") or "Untitled",
                summary=article_record.get("summary") or "",
                source=article_record.get("source") or "Unknown",
                url=article_record.get("url") or "",
                published=article_record.get("published") or "",
                ingested_date=ingested_date,
            )
        except Exception as exc:
            logger.warning("ChromaDB ingestion failed for %s: %s", url, exc)
            article_record["ingested"] = False
        digest.append(article_record)
        logger.info("Processed article: %s", article_record["title"])

    logger.info("Daily digest generation completed - %d successful entries", len([d for d in digest if d["success"]]))
    return digest


if __name__ == "__main__":
    # Simple local test – prints a readable overview of the digest
    result = generate_daily_digest()
    print("=== Daily AI Digest ===")
    for idx, entry in enumerate(result, start=1):
        print(f"\n[{idx}] {entry.get('title')} (Source: {entry.get('source')})")
        print(f"Published: {entry.get('published')}")
        if entry.get('success'):
            print("Summary:")
            print(entry.get('summary'))
            print(f"Provider: {entry.get('provider')}")
        else:
            print("⚠️ Failed –", entry.get('error'))
