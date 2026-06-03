import streamlit as st
import time
import sys

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*__path__.*zoedepth.*"
)

from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()

# Import live backend utility modules from utils/
from utils.scraper import extract_article
from utils.summarizer import universal_summarizer
from utils.linkedin_generator import generate_universal_linkedin_post
from utils.linkedin_generator import generate_linkedin_post
from daily_digest import generate_daily_digest

# Reconfigure stdout to use UTF-8 to prevent encoding warnings on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Page Configuration
st.set_page_config(
    page_title="dAIly digest - AI News Intelligence Platform",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal custom CSS for typography and the brand/badge elements
MINIMAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Apply professional font styling globally */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Brand header without emojis */
.brand-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 8px;
    margin-top: -15px;
}
.brand-title span {
    color: #FF4B91;
}

/* Sentiment badge formatting */
.sentiment-badge {
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
    text-transform: uppercase;
    border: 1px solid currentColor;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
}
.sentiment-positive {
    background-color: rgba(16, 185, 129, 0.1);
    color: #10B981;
}
.sentiment-neutral {
    background-color: rgba(59, 130, 246, 0.1);
    color: #3B82F6;
}
.sentiment-negative {
    background-color: rgba(239, 68, 68, 0.1);
    color: #EF4444;
}
.digest-section-label {
    color: #FF4B91;
    font-weight: 800;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
    margin-bottom: 4px;
}
.digest-summary-panel {
    background-color: rgba(255, 75, 145, 0.06);
    border-left: 4px solid #FF4B91;
    border-radius: 6px;
    padding: 12px 14px;
    margin: 8px 0 12px 0;
}
.digest-date-card {
    text-align: center;
}
.digest-date-card h2 {
    color: #3B82F6;
    margin-bottom: 2px;
}
.digest-date-card p {
    color: #FF4B91;
    font-weight: 700;
    margin-top: 0;
}
</style>
"""
st.markdown(MINIMAL_CSS, unsafe_allow_html=True)


def normalize_sentiment(sentiment_text: str) -> str:
    """Normalize raw sentiment text into a single canonical value."""
    text = (sentiment_text or "").lower().strip()
    if text.startswith("positive") or "positive" in text:
        return "Positive"
    elif text.startswith("negative") or "negative" in text:
        return "Negative"
    return "Neutral"


# Helper function to parse universal news summaries into UI cards
def parse_universal_summary(summary_text: str):
    """
    Parses universal_summarizer() output into:
    - Summary
    - Key Insights
    - Sentiment
    """
    summary = ""
    insights = ""
    sentiment = ""

    normalized_text = (summary_text or "").replace("\r\n", "\n")

    parts = normalized_text.split("### ")
    for part in parts:
        stripped_part = part.strip()
        if stripped_part.startswith("1. Summary"):
            summary = stripped_part.replace("1. Summary", "", 1).strip()
        elif stripped_part.startswith("2. Key Insights"):
            insights = stripped_part.replace("2. Key Insights", "", 1).strip()
        elif stripped_part.startswith("3. Sentiment"):
            sentiment = stripped_part.replace("3. Sentiment", "", 1).strip()

    if not summary:
        summary = summary_text or "No summary was returned."
    if not insights:
        insights = "Key insights were not available. Please review the summary for details."
    
    # Use canonical sentiment exclusively
    sentiment = normalize_sentiment(sentiment)

    return summary, insights, sentiment


# Helper function to parse AI digest summaries into compact intelligence sections
def parse_ai_digest_summary(summary_text: str):
    """
    Parses summarize_article() output into:
    - Concise Summary
    - Key Insights
    - Industry Impact
    - Future Implications
    """
    import re

    normalized_text = (summary_text or "").replace("\r\n", "\n")
    section_pattern = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:\*\*)?(?P<number>[1-4])\.\s*"
        r"(?P<title>Concise Summary|Key Insights|Industry Impact|Future Implications)"
        r"(?:\*\*)?\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE
    )

    sections = {}
    matches = list(section_pattern.finditer(normalized_text))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
        title = match.group("title").lower()
        sections[title] = normalized_text[start:end].strip()

    concise_summary = sections.get("concise summary", "")
    key_insights = sections.get("key insights", "")
    industry_impact = sections.get("industry impact", "")
    future_implications = sections.get("future implications", "")

    if not concise_summary:
        concise_summary = normalized_text.strip() or "No summary was returned for this article."
    if not key_insights:
        key_insights = "Key insights were not available in the generated digest."
    if not industry_impact:
        industry_impact = "Industry impact was not available in the generated digest."
    if not future_implications:
        future_implications = "Future implications were not available in the generated digest."

    return concise_summary, key_insights, industry_impact, future_implications


# ----------------- PAGE 1: HOME PAGE -----------------
def render_home_page():
    st.markdown("<h1 class='brand-title'>d<span>AI</span>ly digest</h1>", unsafe_allow_html=True)
    st.write("### An AI-powered news intelligence platform")
    
    st.write("---")
    
    # Clean product introduction
    st.write(
        "Welcome to dAIly digest, a professional platform designed to aggregate, "
        "analyze, and synthesize the rapidly evolving landscape of artificial intelligence. "
        "By automating manual research workflows, dAIly digest helps AI professionals, "
        "researchers, and technology leaders stay informed with deep, clean, and actionable insights."
    )
    
    st.write("") # Spacer
    
    # Explaining the 4 pillars (no emojis, minimal and professional layout)
    st.subheader("Core Capabilities")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Automated Daily AI Digests")
        st.write(
            "Compile and generate comprehensive summaries of the day's most critical "
            "AI breakthroughs, research paper publications, and corporate news with a single click. "
            "Receive structured intelligence designed for quick scanning and consumption."
        )
        
        st.write("#### Retrieval-Augmented AI Assistance")
        st.write(
            "Engage in conversations with an intelligent assistant trained on deep AI history "
            "and contemporary breakthroughs. Ask questions, compare model paradigms, and "
            "explore the evolutionary lineage of AI theories and architectures."
        )
        
    with col2:
        st.write("#### Universal News Summarizer")
        st.write(
            "Paste any news article URL to instantly scrape, "
            "clean, and structure the content. Extract core summaries and key announcements, "
            "and calculate sentiment indices automatically using active multi-provider LLM routing."
        )
        
        st.write("#### LinkedIn-Ready Content Generation")
        st.write(
            "Bridge the gap between news analysis and professional outreach. "
            "Convert any extracted article summary into a polished, educational, and high-impact "
            "LinkedIn post tailored for professionals, students, and lifelong learners."
        )

    st.write("---")
    
    # Platform Vision section
    st.subheader("Platform Vision")
    st.write(
        "The velocity of AI advancement is outpacing traditional news distribution and consumption structures. "
        "Our vision is to build a noise-filtering layer for the tech industry—empowering professionals with "
        "high-fidelity information streams and automated distribution engines, allowing teams to focus on "
        "integration, development, and strategic execution."
    )


# ----------------- PAGE 2: UNIVERSAL NEWS SUMMARIZER -----------------
def render_summarizer_page():
    st.subheader("Universal News Summarizer")
    st.write("Scrape, analyze, and synthesize any news article URL.")
    
    st.write("---")
    
    # Maintain state keys across selections
    if "summary_generated" not in st.session_state:
        st.session_state.summary_generated = False
    if "linkedin_generated" not in st.session_state:
        st.session_state.linkedin_generated = False
    if "summary_data" not in st.session_state:
        st.session_state.summary_data = {}
        
    # URL input field
    url_input = st.text_input(
        "Article URL",
        placeholder="Paste the URL of an AI news article to summarize and analyze",
    )
    
    # Generate Summary button alignment
    btn_col, _ = st.columns([2, 8])
    with btn_col:
        generate_clicked = st.button("Generate Summary")
        
    if generate_clicked:
        if not url_input.strip():
            st.warning("Please enter a valid article URL to summarize.")
        else:
            # Stage 1: Scrape URL
            with st.spinner("Scraping article content..."):
                scrape_res = extract_article(url_input)
                
            if not scrape_res["success"]:
                st.error(f"Scraping failed: {scrape_res['error']}")
            else:
                # Stage 2: Summarize Content
                with st.spinner("Generating structured news summary..."):
                    summary_res = universal_summarizer(scrape_res["text"])
                    
                if not summary_res["success"]:
                    st.error(f"Summarization failed: {summary_res['error']}")
                else:
                    # Stage 3: Parse and Store Results
                    summary_parsed, insights_parsed, sentiment_parsed = parse_universal_summary(summary_res["summary"])
                    
                    st.session_state.summary_data = {
                        "title": scrape_res["title"] if scrape_res["title"] else "Extracted Article",
                        "summary_raw": summary_res["summary"],
                        "summary": summary_parsed,
                        "insights": insights_parsed,
                        "sentiment": sentiment_parsed,
                        "provider": summary_res["provider"],
                        "linkedin_post": None,
                        "linkedin_provider": None
                    }
                    st.session_state.summary_generated = True
                    st.session_state.linkedin_generated = False  # Reset LinkedIn post state
                
    if st.session_state.summary_generated:
        data = st.session_state.summary_data
        
        st.write("") # Spacer
        st.write(f"### Analysis Results: {data['title']}")
        
        with st.container(border=True):
            st.markdown("### Summary")
            st.write(data["summary"])

        col_left, col_right = st.columns(2)

        with col_left:
            with st.container(border=True):
                st.markdown("**Key Insights**")
                st.write(data["insights"])

        with col_right:
            with st.container(border=True):
                # Use the single canonical sentiment value for all UI layers
                sentiment_label = data.get("sentiment", "Neutral")
                sentiment_class = f"sentiment-{sentiment_label.lower()}"

                st.markdown("**Sentiment**")
                st.markdown(
                    f"<span class='sentiment-badge {sentiment_class}'>{sentiment_label}</span>",
                    unsafe_allow_html=True
                )
                st.write(sentiment_label)
                
        # Subtle provider and metadata lines
        st.caption(f"Routed LLM: {data['provider']} | Status: Successfully Routed")
        
        # Generate LinkedIn Post Section below the summary cards
        st.write("---")
        st.write("### Social Content Distribution")
        st.write("Generate a clean social post tailored for fellow humans.")
        
        post_btn_col, _ = st.columns([2, 8])
        with post_btn_col:
            linkedin_clicked = st.button("Generate LinkedIn Post")
            
        if linkedin_clicked or st.session_state.linkedin_generated:
            if linkedin_clicked and not st.session_state.linkedin_generated:
                with st.spinner("Drafting post copy via routed LLM provider..."):
                    # Generate the post using the live backend module
                    post_res = generate_universal_linkedin_post(data["summary_raw"])
            
                    
                if not post_res["success"]:
                    st.error(f"LinkedIn post generation failed: {post_res['error']}")
                else:
                    st.session_state.summary_data["linkedin_post"] = post_res["linkedin_post"]
                    st.session_state.summary_data["linkedin_provider"] = post_res["provider"]
                    st.session_state.linkedin_generated = True
                    
            if st.session_state.linkedin_generated:
                st.write("")
                with st.container(border=True):
                    st.markdown("**LinkedIn Post Draft**")
                    st.write(data["linkedin_post"])
                    



# ----------------- PAGE 4: DIGEST ASSISTANT (RAG) -----------------
def render_rag_assistant():
    """Render the conversational RAG assistant UI with chat-style interaction."""
    st.subheader("Digest Assistant (RAG)")
    st.write(
        "Ask questions about AI developments, companies, trends, and previously ingested AI intelligence."
    )

    # --- Session state initialisation ---
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []  # List[{"role": str, "content": str, "sources": list|None, "provider": str|None}]
    if "rag_previous_query" not in st.session_state:
        st.session_state.rag_previous_query = ""
    if "rag_previous_topic" not in st.session_state:
        st.session_state.rag_previous_topic = ""

    # --- Render existing chat history ---
    for msg in st.session_state.rag_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show retrieved sources beneath assistant messages.
            if msg["role"] == "assistant" and msg.get("sources"):
                st.write("**Retrieved Sources**")
                for source in msg["sources"]:
                    with st.container(border=True):
                        st.markdown(f"**{source.get('title') or 'Untitled'}**")
                        meta = []
                        if source.get("source"):
                            meta.append(f"Source: {source['source']}")
                        if source.get("published"):
                            meta.append(f"Published: {source['published']}")
                        if source.get("similarity") is not None:
                            meta.append(f"Similarity: {source['similarity']:.2f}")
                        if meta:
                            st.caption(" | ".join(meta))
                        if source.get("url"):
                            st.markdown(f"[Open source]({source['url']})")

            # Show provider badge beneath assistant messages.
            if msg["role"] == "assistant" and msg.get("provider"):
                st.caption(f"Routed LLM: {msg['provider']}")

    # --- Chat input ---
    user_query = st.chat_input("Ask the Digest Assistant...")

    if user_query:
        # Immediately render the user message.
        with st.chat_message("user"):
            st.markdown(user_query)

        # Persist user message.
        st.session_state.rag_chat_history.append({
            "role": "user",
            "content": user_query,
            "sources": None,
            "provider": None,
        })

        # --- Import RAG pipeline (deferred to keep other pages fast) ---
        try:
            from rag_assistant.rag_pipeline import answer_ai_query, detect_topic
        except Exception as exc:
            error_msg = (
                "RAG dependencies are not available yet. Install chromadb, "
                f"sentence-transformers, and langchain, then retry. Details: {exc}"
            )
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.rag_chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "sources": None,
                "provider": None,
            })
            return

        # Build session context for the pipeline.
        session_context = {
            "previous_query": st.session_state.rag_previous_query,
            "previous_topic": st.session_state.rag_previous_topic,
            "chat_history": [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.rag_chat_history
            ],
        }

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant AI intelligence and drafting an answer..."):
                result = answer_ai_query(user_query, session_context=session_context)

            if not result.get("success"):
                error_text = f"Assistant failed: {result.get('error') or 'Unknown error'}"
                st.error(error_text)
                st.session_state.rag_chat_history.append({
                    "role": "assistant",
                    "content": error_text,
                    "sources": None,
                    "provider": None,
                })
            else:
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                provider = result.get("provider")

                st.markdown(answer)

                # Display retrieved sources.
                if sources:
                    st.write("**Retrieved Sources**")
                    for source in sources:
                        with st.container(border=True):
                            st.markdown(f"**{source.get('title') or 'Untitled'}**")
                            meta = []
                            if source.get("source"):
                                meta.append(f"Source: {source['source']}")
                            if source.get("published"):
                                meta.append(f"Published: {source['published']}")
                            if source.get("similarity") is not None:
                                meta.append(f"Similarity: {source['similarity']:.2f}")
                            if meta:
                                st.caption(" | ".join(meta))
                            if source.get("url"):
                                st.markdown(f"[Open source]({source['url']})")

                if provider:
                    st.caption(f"Routed LLM: {provider}")

                # Persist assistant message with sources.
                st.session_state.rag_chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "provider": provider,
                })

                # Update topic continuity state.
                st.session_state.rag_previous_query = user_query
                st.session_state.rag_previous_topic = (
                    detect_topic(user_query) or st.session_state.rag_previous_topic
                )



# ----------------- PAGE 4: DAILY DIGEST -----------------
def render_daily_digest():
    """Render the Daily AI Digest UI page.
    Uses the backend generate_daily_digest() to fetch, scrape, and summarize AI news.
    """
    from datetime import datetime
    import html

    st.subheader("Daily AI Digest")
    st.write("AI-powered daily intelligence briefing of the latest developments in artificial intelligence.")

    current_date = datetime.now()
    with st.container(border=True):
        st.markdown(
            (
                "<div class='digest-date-card'>"
                f"<h2>{current_date.strftime('%A, %B %d, %Y')}</h2>"
                "<p>AI Intelligence Briefing</p>"
                "</div>"
            ),
            unsafe_allow_html=True
        )

    # Initialize session state keys
    if "daily_digest" not in st.session_state:
        st.session_state.daily_digest = []
    if "digest_generated" not in st.session_state:
        st.session_state.digest_generated = False

    article_count = st.number_input(
        "Number of AI articles to fetch",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    generate_clicked = st.button("Generate Today's Digest")
    if generate_clicked:
        with st.spinner("Generating today's AI digest..."):
            digest = generate_daily_digest(max_articles=article_count)
        st.session_state.daily_digest = digest
        st.session_state.digest_generated = True

    if st.session_state.digest_generated and st.session_state.daily_digest:
        for idx, article in enumerate(st.session_state.daily_digest, start=1):
            with st.container(border=True):
                st.markdown(f"### {idx}. {article.get('title') or 'Untitled'}")

                meta = []
                if article.get('source'):
                    meta.append(f"Source: {article.get('source')}")
                if article.get('provider'):
                    meta.append(f"Provider: {article.get('provider')}")
                if article.get('published'):
                    meta.append(f"Published: {article.get('published')}")
                if meta:
                    st.caption(" | ".join(meta))

                if article.get('url'):
                    st.markdown(f"[Read full article]({article.get('url')})")

                if article.get('success'):
                    (
                        concise_summary,
                        key_insights,
                        industry_impact,
                        future_implications
                    ) = parse_ai_digest_summary(article.get('summary'))

                    st.markdown("<div class='digest-section-label'>Concise Summary</div>", unsafe_allow_html=True)
                    safe_concise_summary = html.escape(concise_summary).replace("\n", "<br>")
                    st.markdown(
                        f"<div class='digest-summary-panel'>{safe_concise_summary}</div>",
                        unsafe_allow_html=True
                    )

                    lower_left, lower_right = st.columns(2)
                    with lower_left:
                        st.markdown("<div class='digest-section-label'>Key Insights</div>", unsafe_allow_html=True)
                        st.write(key_insights)

                    with lower_right:
                        st.markdown("<div class='digest-section-label'>Industry Impact</div>", unsafe_allow_html=True)
                        st.write(industry_impact)

                    st.markdown("<div class='digest-section-label'>Future Implications</div>", unsafe_allow_html=True)
                    st.write(future_implications)

                    # --- LinkedIn Post Generation (per article) ---
                    st.markdown("---")
                    linkedin_key = f"linkedin_digest_{idx}"
                    state_key = f"linkedin_digest_post_{idx}"

                    if st.button("Generate LinkedIn Post", key=linkedin_key):
                        with st.spinner("Generating LinkedIn post..."):
                            post_result = generate_linkedin_post(article.get("summary", ""))
                        if post_result["success"]:
                            st.session_state[state_key] = post_result["linkedin_post"]
                        else:
                            st.session_state[state_key] = None
                            st.error(f"LinkedIn generation failed: {post_result['error']}")

                    if st.session_state.get(state_key):
                        st.markdown("**LinkedIn Post**")
                        st.text_area(
                            "Copy your LinkedIn post below",
                            value=st.session_state[state_key],
                            height=250,
                            disabled=True,
                            key=f"linkedin_digest_textarea_{idx}"
                        )
                else:
                    st.warning(f"Failed to process article: {article.get('error') or 'Unknown error'}")
    elif st.session_state.digest_generated:
        st.info("No digest articles were generated. Please try again later.")



# ----------------- MAIN APP ENTRYPOINT -----------------
def main():
    # Sidebar Navigation Selection
    st.sidebar.title("dAIly digest")
    page_selection = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Daily AI Digest",
            "Digest Assistant (RAG)",
            "Universal News Summarizer"
        ]
    )
    
    # Route selection to corresponding page rendering functions
    if page_selection == "Home":
        render_home_page()
    elif page_selection == "Daily AI Digest":
        render_daily_digest()
    elif page_selection == "Digest Assistant (RAG)":
        render_rag_assistant()
    elif page_selection == "Universal News Summarizer":
        render_summarizer_page()


if __name__ == "__main__":
    main()
