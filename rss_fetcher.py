# rss_fetcher.py
"""Utility module to fetch the latest AI news articles from curated RSS feeds.

The module provides a single public function `fetch_latest_ai_news` which returns a list of
article metadata dictionaries selected via balanced round-robin source traversal.

Features
--------
- Uses the `feedparser` library for RSS parsing.
- Curated list of AI-focused RSS sources with strict ordering for round-robin selection.
- AI relevance filtering: only articles scoring >= 2 keyword matches are retained.
- Round-robin source balancing: articles are drawn one-per-feed per pass to maximise diversity.
- Graceful handling of unavailable or malformed feeds – a broken feed is logged and skipped.
- Configurable maximum number of articles to return.
- Production-ready structure: type hints, detailed docstrings, configurable logger.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Mapping, Optional

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
        Human-readable title of the article.
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
        """Return a JSON-serialisable dictionary.

        The ``published`` datetime is ISO-8601 formatted (UTC).
        """
        return {
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "published": self.published.isoformat() + "Z",
        }

# ---------------------------------------------------------------------------
# Curated RSS feed catalogue
# Order is preserved and drives round-robin article selection.
# ---------------------------------------------------------------------------
_FEED_CATALOGUE: Mapping[str, str] = {
    "TLDR AI": "https://tldr.tech/api/r ss/ai",
    "AI Feed": "https://aifeed.dev/feed.xml",
    "Wired AI:": "https://www.wired.com/feed/tag/ai/latest/rss",
    # "VentureBeat AI": "https://venturebeat.com/category/ai/feed",
    "OpenAI": "https://openai.com/news/rss.xml",
    "NVIDIA": "https://nvidianews.nvidia.com/rss.xml",
}

# ---------------------------------------------------------------------------
# AI relevance filtering
# ---------------------------------------------------------------------------
AI_RELEVANCE_KEYWORDS: List[str] = [
    "ai",
    "artificial intelligence",
    "machine learning",
    "llm",
    "gpt",
    "chatgpt",
    "openai",
    "anthropic",
    "claude",
    "transformer",
    "foundation model",
    "generative ai",
    "multimodal",
    "rag",
    "vector database",
    "neural network",
    "gpu",
    "nvidia",
    "diffusion",
    "deep learning",
    "ai infrastructure",
    "agentic ai",
]


def is_ai_relevant(title: str, summary: str) -> bool:
    """Determine whether an article is relevant to the AI ecosystem.

    Combines the article title and summary into a single lowercase text blob,
    then counts how many of the curated ``AI_RELEVANCE_KEYWORDS`` appear in it.
    An article is considered relevant only when the cumulative keyword score
    reaches 2 or more, preventing weak single-keyword coincidences from
    slipping through (e.g. a gadget article that mentions "GPU" once).

    The check is:
    - Lightweight   – pure string operations, no external calls.
    - Deterministic – identical inputs always produce identical outputs.
    - Fast           – O(N·K) where N = text length and K = keyword count.
    - Explainable   – each keyword hit contributes exactly +1 to the score.

    Parameters
    ----------
    title: str
        Article headline.
    summary: str
        Short description or excerpt from the RSS entry (may be empty).

    Returns
    -------
    bool
        ``True`` if the combined score is >= 2; ``False`` otherwise.
    """
    combined = (title + " " + summary).lower()
    score = sum(1 for keyword in AI_RELEVANCE_KEYWORDS if keyword in combined)
    return score >= 2


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
        Human-readable name of the originating feed.

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
    """Retrieve, parse, and AI-filter a single RSS feed.

    Articles that do not pass :func:`is_ai_relevant` are discarded before being
    returned, so that only genuinely AI-ecosystem content flows into the
    selection and scraping pipeline.

    Parameters
    ----------
    url: str
        Feed URL.
    source_name: str
        Human-readable identifier for logging.

    Returns
    -------
    List[Article]
        AI-relevant articles extracted from the feed; an empty list if the
        feed cannot be processed or yields no relevant content.
    """
    logger.info("Fetching feed for %s from %s", source_name, url)
    try:
        # First attempt using feedparser's built-in fetch (may fail on SSL issues).
        parsed = feedparser.parse(url)
        if parsed.bozo and isinstance(parsed.bozo_exception, Exception):
            # If bozo indicates a parsing error (often SSL), fall back to requests.
            raise parsed.bozo_exception
        # Successful parse – fall through to article extraction below.
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
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
                verify=False,
            )
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

    raw_count = len(parsed.entries)
    logger.info("Parsed %d entries from %s", raw_count, source_name)

    articles: List[Article] = []
    for entry in parsed.entries:
        article = _parse_entry(entry, source_name)
        if article is None:
            continue

        # AI relevance gate – use the RSS summary/description if available.
        entry_summary = (
            entry.get("summary", "")
            or entry.get("description", "")
            or ""
        )
        if not is_ai_relevant(article.title, entry_summary):
            logger.debug(
                "Filtered out (low AI relevance): '%s' from %s", article.title, source_name
            )
            continue

        articles.append(article)

    logger.info(
        "%d / %d articles passed AI relevance filtering for %s",
        len(articles),
        raw_count,
        source_name,
    )
    return articles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_latest_ai_news(max_articles: int = 3) -> List[Mapping[str, str]]:
    """Fetch the most recent AI news items from curated RSS feeds.

    Articles are selected using a **round-robin** strategy that iterates over
    the ordered ``_FEED_CATALOGUE`` feeds in sequence, picking one article per
    feed per pass until ``max_articles`` is reached or all feeds are exhausted.

    This guarantees balanced source diversity regardless of publication volume:
    a high-volume feed like VentureBeat cannot crowd out a lower-volume source
    like OpenAI.

    Selection example for max_articles=8 (6 feeds):
        Pass 1: TLDR AI → AI Feed → Wired AI → VentureBeat AI → OpenAI → NVIDIA
        Pass 2: TLDR AI (2nd article) → AI Feed (2nd article)

    Parameters
    ----------
    max_articles: int, optional
        Upper bound on the number of articles to return. Defaults to ``3``.

    Returns
    -------
    List[Mapping[str, str]]
        A list of dictionaries with the keys ``title``, ``source``, ``url`` and
        ``published`` (ISO-8601 UTC string).
    """
    if max_articles <= 0:
        logger.warning("max_articles must be a positive integer; received %s", max_articles)
        return []

    # Step 1 – fetch and AI-filter each feed individually, preserving catalogue order.
    feed_buckets_raw: Dict[str, List[Article]] = {}
    for source, feed_url in _FEED_CATALOGUE.items():
        feed_buckets_raw[source] = _fetch_feed(feed_url, source)
        
    # Find the most recent date with available articles
    all_articles = [a for bucket in feed_buckets_raw.values() for a in bucket]
    if not all_articles:
        logger.info("No AI-relevant articles found across any feeds.")
        return []
        
    latest_date_str = max(a.published.strftime("%Y-%m-%d") for a in all_articles)
    logger.info("Most recent available articles found on: %s", latest_date_str)
    
    # Filter the buckets to only include articles from the latest available date
    feed_buckets: Dict[str, List[Article]] = {}
    for source, bucket in feed_buckets_raw.items():
        feed_buckets[source] = [a for a in bucket if a.published.strftime("%Y-%m-%d") == latest_date_str]

    # Step 2 – round-robin selection across feeds.
    selected: List[Article] = []
    seen_urls: set = set()
    seen_titles: set = set()

    feed_names = list(_FEED_CATALOGUE.keys())       # preserves insertion order
    cursors: Dict[str, int] = {name: 0 for name in feed_names}

    while len(selected) < max_articles:
        made_progress = False

        for source in feed_names:
            if len(selected) >= max_articles:
                break

            bucket = feed_buckets[source]
            cursor = cursors[source]

            # Advance cursor past duplicates within this feed.
            while cursor < len(bucket):
                candidate = bucket[cursor]
                cursor += 1

                if candidate.url in seen_urls:
                    logger.debug("Skipped duplicate URL: %s", candidate.url)
                    continue
                if candidate.title in seen_titles:
                    logger.debug("Skipped duplicate title: %s", candidate.title)
                    continue

                # Valid candidate – select it.
                selected.append(candidate)
                seen_urls.add(candidate.url)
                seen_titles.add(candidate.title)
                cursors[source] = cursor
                made_progress = True
                logger.info(
                    "Selected article from %s: '%s'", source, candidate.title
                )
                break
            else:
                # Feed exhausted – update cursor so we don't recheck.
                cursors[source] = cursor

        if not made_progress:
            # All feeds exhausted before reaching max_articles.
            logger.info(
                "All feeds exhausted after selecting %d articles (requested %d)",
                len(selected),
                max_articles,
            )
            break

    logger.info(
        "Returning %d round-robin selected articles (requested %d)",
        len(selected),
        max_articles,
    )
    return [article.to_dict() for article in selected]


# ---------------------------------------------------------------------------
# Module self-test (run with ``python rss_fetcher.py``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    fetched = fetch_latest_ai_news()
    print(json.dumps(fetched, indent=2, ensure_ascii=False))
