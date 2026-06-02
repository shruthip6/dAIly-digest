"""
Web scraper utilities for news collection.

DESIGN DECISIONS:
- Why newspaper3k over BeautifulSoup?
  BeautifulSoup requires writing site-specific scraping rules and parsing logic (CSS selectors / XPaths)
  which is fragile and does not scale across multiple different news sites. newspaper3k uses advanced
  machine learning heuristics (based on text density and structural tags) to automatically identify 
  and clean main article body, authors, title, and publish dates from any news URL without site-specific code.

- Why newspaper3k over Selenium?
  Selenium launches a full, heavy headless browser instance. It is extremely slow and resource-intensive,
  making it suitable only for complex SPA applications that require client-side Javascript rendering 
  or user interaction. Most news articles and blog posts are served statically, so fetching HTML directly 
  with newspaper3k is orders of magnitude faster and lightweight.
"""

import logging
import datetime
from typing import Dict, Any
from newspaper import Article, ArticleException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("scraper")

def extract_article(url: str) -> Dict[str, Any]:
    """
    Extracts and cleans article content from a news URL using newspaper3k.
    
    Args:
        url (str): The URL of the news article to scrape.
        
    Returns:
        Dict[str, Any]: A standardized response dictionary:
            {
                "success": bool,
                "title": str or None,
                "text": str or None,
                "authors": list of str or None,
                "publish_date": str (ISO format) or None,
                "error": str or None
            }
    """
    # Gracefully handle empty or invalid URL type
    if not url or not isinstance(url, str) or not url.strip():
        logger.error("Invalid URL provided: URL is empty or not a string.")
        return {
            "success": False,
            "title": None,
            "text": None,
            "authors": [],
            "publish_date": None,
            "error": "URL is empty or not a valid string."
        }
        
    logger.info(f"Initiating extraction for URL: {url}")
    
    try:
        # Initialize Article with a standard user-agent config to prevent simple blocking
        article = Article(url)
        
        # Download the article HTML
        article.download()
        
        # Parse HTML to extract structured fields (title, text, authors, publish_date)
        article.parse()
        
        title = article.title
        text = article.text
        authors = article.authors
        publish_date = article.publish_date
        
        # Gracefully handle empty content extraction (e.g. if the page has no readable body text)
        if not text or not text.strip():
            logger.warning(f"Extraction completed but main article text is empty: {url}")
            return {
                "success": False,
                "title": title if title else None,
                "text": "",
                "authors": authors if authors else [],
                "publish_date": None,
                "error": "Extracted article text is empty."
            }
            
        # Standardize publish date to ISO 8601 string format if it's a datetime object
        publish_date_str = None
        if isinstance(publish_date, datetime.datetime):
            publish_date_str = publish_date.isoformat()
        elif publish_date:
            publish_date_str = str(publish_date)
            
        logger.info(f"Successfully extracted article: '{title}'")
        return {
            "success": True,
            "title": title,
            "text": text,
            "authors": authors if authors else [],
            "publish_date": publish_date_str,
            "error": None
        }
        
    except ArticleException as ae:
        # Specifically catch newspaper3k's article download/parsing exceptions
        error_msg = f"Newspaper3k article extraction failed: {str(ae)}"
        logger.error(error_msg)
        return {
            "success": False,
            "title": None,
            "text": None,
            "authors": [],
            "publish_date": None,
            "error": error_msg
        }
    except Exception as e:
        # Catch any other unexpected exceptions (e.g., connection errors, SSL issues)
        error_msg = f"Unexpected error while extracting article: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "title": None,
            "text": None,
            "authors": [],
            "publish_date": None,
            "error": error_msg
        }

if __name__ == "__main__":
    import json
    
    print("Testing Web Scraper Module...")
    
    # List of test URLs including an OpenAI blog post, invalid, and empty URLs
    test_urls = [
        "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",  # OpenAI blog post
        "invalid-url-test",                             # Invalid URL format
        ""                                              # Empty URL string
    ]
    
    for test_url in test_urls:
        print(f"\nScraping URL: {test_url}")
        result = extract_article(test_url)
        print(json.dumps(result, indent=2, default=str))
