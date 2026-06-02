# rss_fetcher.py
"""Utility module to fetch the latest AI news articles from curated RSS feeds.

The module provides a single public function `fetch_latest_ai_news` which returns a list of
article metadata dictionaries sorted by publication date (most recent first).

Features
--------
- Uses the `feedparser` library for RSS parsing.
- Curated list of AI‑focused RSS sources (OpenAI, Anthropic, DeepMind, Hugging Face, NVIDIA).
- Graceful handling of unavailable or malformed feeds – a broken feed is logged and skipped.
- Configurable maximum number of articles to return.
- Production‑ready structure: type hints, detailed docstrings, configurable logger.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Mapping, Optional

import feedparser

# ---------------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Configure a simple console logger only once (idempotent).
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Article:
    """Immutable representation of a news article.

    Attributes
    ----------
    title: str
        Human‑readable title of the article.
    source: str
        Human readable name of the RSS feed/source.
    url: str
        Direct link to the article.
    published: datetime
        Publication timestamp (UTC). If the feed does not provide a date, the current
        UTC time is used as a fallback.
    """

    title: str
    source: str
    url: str
    published: datetime

    def to_dict(self) -> Mapping[str, str]:
        """Return a JSON‑serialisable dictionary.

        The ``published`` datetime is ISO‑8601 formatted (UTC).
        """
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published": self.published.isoformat() + "Z",
        }

# ---------------------------------------------------------------------------
# Curated RSS feed catalogue
# ---------------------------------------------------------------------------
_FEED_CATALOGUE: Mapping[str, str] = {
    "OpenAI": "https://openai.com/news/rss.xml",
    # Anthropic does not expose an official RSS – use RSSHub as a reliable proxy.
    "Anthropic": "https://rsshub.app/anthropic/blog",
    # DeepMind (Google) – no official feed; RSSHub provides a scraped version.
    "DeepMind": "https://rsshub.app/deepmind/blog",
    "Hugging Face": "https://huggingface.co/blog/rss",
    # NVIDIA – primary newsroom feed.
    "NVIDIA": "https://nvidianews.nvidia.com/rss.xml",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _parse_entry(entry: dict, source_name: str) -> Optional[Article]:
    """Convert a raw FeedParser entry into an :class:`Article`.

    Parameters
    ----------
    entry: dict
        Single entry object returned by ``feedparser``.
    source_name: str
        Human‑readable name of the originating feed.

    Returns
    -------
    Optional[Article]
        ``None`` if mandatory fields are missing; otherwise a populated ``Article``.
    """
    try:
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        if not title or not link:
            logger.debug(
                "Skipping entry from %s due to missing title or link", source_name
            )
            return None

        # Prefer the standard ``published_parsed`` field; fall back to ``updated_parsed``.
        parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed_time:
            published_dt = datetime(*parsed_time[:6])
        else:
            # No date information – use current UTC as a safe placeholder.
            published_dt = datetime.utcnow()
            logger.debug(
                "Entry '%s' from %s lacks a timestamp; using current UTC", title, source_name
            )

        return Article(
            title=title,
            source=source_name,
            url=link,
            published=published_dt,
        )
    except Exception as exc:
        logger.exception(
            "Unexpected error while parsing entry from %s: %s", source_name, exc
        )
        return None

def _fetch_feed(url: str, source_name: str) -> List[Article]:
    """Retrieve and parse a single RSS feed.

    Parameters
    ----------
    url: str
        Feed URL.
    source_name: str
        Human‑readable identifier for logging.

    Returns
    -------
    List[Article]
        Articles extracted from the feed; an empty list if the feed cannot be processed.
    """
    logger.info("Fetching feed for %s from %s", source_name, url)
    try:
        # First attempt using feedparser's built‑in fetch (may fail on SSL issues)
        parsed = feedparser.parse(url)
        if parsed.bozo and isinstance(parsed.bozo_exception, Exception):
            # If bozo indicates a parsing error (often SSL), fall back to requests with verification disabled.
            raise parsed.bozo_exception
        # Successful parse, return articles below.
    except Exception as primary_exc:
        logger.warning(
            "Primary fetch/parsing failed for %s (%s): %s – attempting fallback with insecure request",
            source_name,
            url,
            primary_exc,
        )
        try:
            import requests
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
        except Exception as fallback_exc:
            logger.error(
                "Fallback fetch failed for %s (%s): %s",
                source_name,
                url,
                fallback_exc,
            )
            return []
        articles: List[Article] = []
        for entry in parsed.entries:
            article = _parse_entry(entry, source_name)
            if article:
                articles.append(article)
        logger.info("Fetched %d articles from %s", len(articles), source_name)
        return articles
    except Exception as exc:
        logger.exception("Error fetching feed %s (%s): %s", source_name, url, exc)
        return []

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_latest_ai_news(max_articles: int = 3) -> List[Mapping[str, str]]:
    """Fetch the most recent AI news items from curated RSS feeds.

    The function aggregates articles from a predefined set of sources, sorts them by
    publication date (newest first) and returns up to ``max_articles`` entries.

    Parameters
    ----------
    max_articles: int, optional
        Upper bound on the number of articles to return. Defaults to ``3``.

    Returns
    -------
    List[Mapping[str, str]]
        A list of dictionaries with the keys ``title``, ``source``, ``url`` and
        ``published`` (ISO‑8601 UTC string).
    """
    if max_articles <= 0:
        logger.warning("max_articles must be a positive integer; received %s", max_articles)
        return []

    all_articles: List[Article] = []
    for source, feed_url in _FEED_CATALOGUE.items():
        articles = _fetch_feed(feed_url, source)
        all_articles.extend(articles)

    # Sort globally by publication timestamp descending.
    all_articles.sort(key=lambda a: a.published, reverse=True)

    # Slice to the requested limit.
    selected = all_articles[:max_articles]
    logger.info("Returning %d aggregated articles (requested %d)", len(selected), max_articles)
    return [article.to_dict() for article in selected]

# ---------------------------------------------------------------------------
# Module self‑test (run with ``python -m rss_fetcher``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    fetched = fetch_latest_ai_news()
    print(json.dumps(fetched, indent=2, ensure_ascii=False))
