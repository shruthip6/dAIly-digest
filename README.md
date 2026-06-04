# dAIly digest

An AI-powered intelligence platform for real-time news aggregation, AI ecosystem analysis, conversational retrieval, and automated content generation.

---

# Overview

dAIly digest is designed to simplify how professionals consume and interact with rapidly evolving information streams.

The platform combines:

* automated AI news ingestion,
* intelligent summarization,
* conversational Retrieval-Augmented Generation (RAG),
* metadata-aware retrieval,
* sentiment analysis,
* and LinkedIn-ready content generation

into a single unified workflow.

The system transforms fragmented news into structured, searchable, and conversational intelligence.

---

# Core Features

## 1. Daily AI Digest

Automatically fetches and summarizes the latest AI developments from curated RSS feeds.

### Current Feed Sources

* TLDR AI
* AI Feed
* Wired AI
* OpenAI
* NVIDIA

### Key Capabilities

* Live RSS ingestion
* AI relevance filtering
* Current-date article filtering
* Balanced round-robin source selection
* AI-powered structured summarization
* One-click LinkedIn post generation
* Automatic ChromaDB ingestion

---

## 2. AI Digest Assistant (RAG)

A conversational Retrieval-Augmented AI assistant capable of answering contextual questions about:

* AI companies
* AI model evolution
* infrastructure trends
* previously ingested news
* foundational AI knowledge

### Assistant Capabilities

* Semantic retrieval using ChromaDB
* Conversational session memory
* Metadata-aware retrieval
* Temporal awareness
* Source-aware retrieval
* Follow-up question understanding
* Hybrid grounded generation

### Example Queries

* “What OpenAI news was ingested yesterday?”
* “What recent NVIDIA developments were discussed?”
* “What are transformers?”
* “How did they become popular?”
* “Give me the article link for the OpenAI model update yesterday.”

---

## 3. Universal News Summarizer

Summarizes ANY news article URL — not limited to AI.

### Features

* Automatic article scraping
* Content cleaning
* Structured summarization
* Insight extraction
* Sentiment analysis
* Multi-provider LLM routing
* LinkedIn-ready content generation

### Output Structure

* Concise Summary
* Key Insights
* Sentiment
* Provider Used
* Optional LinkedIn Post

---

## 4. LinkedIn Post Generation

Transforms summarized news into concise, professional LinkedIn-ready content.

### Supported Flows

* Daily AI Digest articles
* Universal News Summarizer articles

### Output Characteristics

* Professional tone
* Educational framing
* Strategic industry perspective
* Minimal hype
* Readable formatting

---

# System Architecture

## High-Level Flow

```text
RSS Feeds / User URL
        ↓
Content Fetching & Scraping
        ↓
Content Cleaning
        ↓
LLM Summarization
        ↓
Structured Output Generation
        ↓
Optional:
- LinkedIn Post Generation
- ChromaDB Ingestion
        ↓
RAG Assistant Retrieval
        ↓
Conversational AI Responses
```

---

# Backend Architecture

## 1. RSS Ingestion Layer

### File

`rss_fetcher.py`

### Responsibilities

* Fetch RSS feeds
* Parse entries
* Filter current-date articles
* AI relevance filtering
* Round-robin source balancing
* Deduplicate URLs

### Selection Strategy

The system prioritizes:

* source diversity,
* current-day relevance,
* and AI ecosystem coverage.

---

## 2. Scraping Layer

### File

`utils/scraper.py`

### Responsibilities

* Extract article content from URLs
* Clean HTML noise
* Handle parsing failures
* Provide fallback-safe extraction

---

## 3. Summarization Layer

### File

`summarizer.py`

### Responsibilities

Generate:

* concise summaries
* insights
* industry implications
* future implications
* sentiment analysis

Uses:

* prompt-engineered structured generation
* multi-provider LLM routing

---

## 4. LLM Routing Layer

### File

`utils/llm_router.py`

### Supported Providers

* OpenRouter
* Cohere
* Groq

### Responsibilities

* Provider failover
* Prompt routing
* Unified response structure
* Standardized error handling

### Why Routing Matters

Ensures:

* resilience,
* provider fallback,
* and uninterrupted generation.

---

## 5. LinkedIn Generation Layer

### File

`utils/linkedin_generator.py`

### Responsibilities

Convert summaries into:

* concise,
* professional,
* LinkedIn-ready posts.

Supports:

* Daily AI Digest flow
* Universal Summarizer flow

---

## 6. Vector Database Layer

### Technology

ChromaDB

### Responsibilities

Store:

* AI digest summaries
* foundational AI knowledge
* embeddings
* metadata

### Stored Metadata

* title
* source
* published date
* URL

---

## 7. Embedding Layer

### Model

`sentence-transformers/all-MiniLM-L6-v2`

### Why This Model?

Chosen because it is:

* lightweight
* fast
* optimized for semantic similarity
* efficient for CPU inference
* suitable for local embedding generation

### Embedding Flow

```text
Text Chunk
    ↓
Embedding Model
    ↓
Vector Representation
    ↓
Stored in ChromaDB
```

---

# RAG Architecture

## Retrieval Pipeline

### File

`rag_assistant/rag_pipeline.py`

### Retrieval Flow

```text
User Query
    ↓
Intent Detection
    ↓
Temporal Context Detection
    ↓
Source Detection
    ↓
Conversation Context Augmentation
    ↓
Semantic Retrieval
    ↓
Metadata Filtering
    ↓
LLM Grounded Response Generation
```

---

## Metadata-Aware Retrieval

The assistant supports:

* date-aware retrieval
* source-aware retrieval
* semantic retrieval

### Examples

* “OpenAI news yesterday”
* “Recent Anthropic updates”
* “What articles were stored today?”

---

## Conversational Memory

The assistant maintains:

* lightweight session memory
* follow-up understanding
* contextual continuity

### Example

```text
Q: What are transformers?
Q: How did they become popular?
```

The assistant resolves:
“they” → “transformers”

---

# Sentiment Analysis

The Universal Summarizer includes:

* sentiment classification
* explanation generation
* badge-based visualization

### Canonical Labels

* Positive
* Neutral
* Negative

### Features

* label normalization
* explanation consistency
* synchronized badge rendering

---

# Frontend

## Framework

Streamlit

## UI Sections

* Home
* Daily AI Digest
* AI Digest Assistant
* Universal News Summarizer

### Design Philosophy

* minimal
* structured
* professional
* information-focused

---

# Current Technical Highlights

* Multi-provider LLM routing
* Conversational RAG
* Metadata-aware retrieval
* Temporal AI news awareness
* AI relevance filtering
* Balanced RSS ingestion
* Local embedding generation
* ChromaDB vector retrieval
* Session conversational continuity
* Structured AI summarization

---

# Technologies Used

## Frontend

* Streamlit

## Backend

* Python

## AI/LLM

* OpenRouter - deepseek/deepseek-chat
* Cohere - command-a-03-2025
* Groq - llama-3.3-70b-versatile

## Retrieval

* ChromaDB
* Sentence Transformers - all-MiniLM-L6-v2

## NLP

* HuggingFace Transformers 

## Data Processing

* feedparser
* newspaper
  

---

# Future Scope

Potential future enhancements include:

* personalized RSS profiles
* user-specific digest preferences
* advanced sentiment analytics
* article clustering
* semantic topic tracking
* dashboard analytics
* citation-aware responses
* long-term memory
* agentic workflows

---

# Platform Vision

Transforming real-time information into conversational intelligence.

---

#

Built as an AI intelligence and retrieval platform focused on combining:

* news aggregation,
* structured summarization,
* conversational AI,
* and retrieval-based intelligence systems.
