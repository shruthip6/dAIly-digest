import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("summarizer")

# Import the LLM routing orchestrator from llm_router.py
# Bound as generate_response to satisfy the import architectural requirement
try:
    from utils.llm_router import route_prompt as generate_response
except ImportError:
    from llm_router import route_prompt as generate_response

def summarize_article(article_text: str) -> Dict[str, Any]:
    """
    Generates an AI-powered summary for the provided article text using routed LLM providers.
    
    Args:
        article_text (str): The raw text of the article.
        
    Returns:
        Dict[str, Any]: A standardized response dictionary:
            {
                "success": bool,
                "provider": str or None,
                "summary": str or None,
                "error": str or None
            }
    """
    # Gracefully handle empty or invalid article text inputs
    if not article_text or not isinstance(article_text, str) or not article_text.strip():
        logger.error("Invalid input: Article text is empty or not a string.")
        return {
            "success": False,
            "provider": None,
            "summary": None,
            "error": "Article text is empty or not a valid string."
        }
        
    # Truncate excessively long article text to manage context length, speed, and cost.
    # 12,000 characters is roughly 2,000-3,000 words, which is ideal for a summary prompt.
    MAX_CHARACTERS = 12000
    if len(article_text) > MAX_CHARACTERS:
        logger.info(f"Article text length ({len(article_text)} chars) exceeds maximum. Truncating to {MAX_CHARACTERS} chars.")
        truncated_text = article_text[:MAX_CHARACTERS] + "\n... [Content Truncated for Length] ..."
    else:
        truncated_text = article_text

    logger.info("Generating summary for article text...")
    
    # Role priming anchors the model as an AI industry analyst for more domain-specific synthesis.
    # Strict markdown headers improve downstream rendering stability across Groq, OpenRouter, and Cohere.
    # Task decomposition separates summary, insight, impact, and future analysis without exposing reasoning.
    prompt = (
        "You are a senior AI industry intelligence analyst writing for AI engineers, founders, "
        "researchers, and technology leaders.\n\n"
        "Analyze the article as an AI ecosystem signal. Focus on what changed, why it matters, "
        "who is affected, and what credible future direction it suggests.\n\n"
        "Output rules:\n"
        "- Use the exact markdown section headers below, in the exact order.\n"
        "- Do not add, remove, rename, or reorder sections.\n"
        "- Keep the writing concise, specific, and information-dense.\n"
        "- Avoid hype, motivational language, filler phrases, and generic observations.\n"
        "- Use clear bullets where they improve scanning.\n\n"
        "Required output structure:\n\n"
        "### 1. Concise Summary\n"
        "Write 2-3 sentences capturing the core AI news, the main actors, and the concrete development.\n\n"
        "### 2. Key Insights\n"
        "Provide 3 bullets with the most important technical, research, market, or business signals. "
        "Prefer specific facts, numbers, product changes, or strategic moves from the article.\n\n"
        "### 3. Industry Impact\n"
        "Write one short paragraph explaining implications for AI infrastructure, developer workflows, "
        "business adoption, competition, regulation, or the research ecosystem as relevant.\n\n"
        "### 4. Future Implications\n"
        "Write 1-2 sentences on likely next steps, second-order effects, or longer-term strategic significance. "
        "Stay grounded in the article's evidence.\n\n"
        "Article Content:\n"
        f"{truncated_text}\n"
    )
    
    try:
        # Route prompt to the active/fallback LLM providers
        result = generate_response(prompt)
        
        if result["success"]:
            logger.info(f"Successfully generated summary using provider: {result['provider']}")
            summary_text = result["response"]
            if summary_text:
                # Post-process to convert any literal "\\n" string sequences into actual newlines
                summary_text = summary_text.replace("\\n", "\n")
            return {
                "success": True,
                "provider": result["provider"],
                "summary": summary_text,
                "error": None
            }
        else:
            error_msg = f"LLM Routing failed to generate response: {result['error']}"
            logger.error(error_msg)
            return {
                "success": False,
                "provider": None,
                "summary": None,
                "error": error_msg
            }
            
    except Exception as e:
        error_msg = f"Unexpected error in summarizer module: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "provider": None,
            "summary": None,
            "error": error_msg
        }

def universal_summarizer(article_text: str) -> Dict[str, Any]:
    """
    Generates a professional summary for any news article using routed LLM providers.

    Args:
        article_text (str): The raw text of the article.

    Returns:
        Dict[str, Any]: A standardized response dictionary:
            {
                "success": bool,
                "provider": str or None,
                "summary": str or None,
                "error": str or None
            }
    """
    if not article_text or not isinstance(article_text, str) or not article_text.strip():
        logger.error("Invalid input: Article text is empty or not a string.")
        return {
            "success": False,
            "provider": None,
            "summary": None,
            "error": "Article text is empty or not a valid string."
        }

    MAX_CHARACTERS = 12000
    if len(article_text) > MAX_CHARACTERS:
        logger.info(f"Article text length ({len(article_text)} chars) exceeds maximum. Truncating to {MAX_CHARACTERS} chars.")
        truncated_text = article_text[:MAX_CHARACTERS] + "\n... [Content Truncated for Length] ..."
    else:
        truncated_text = article_text

    logger.info("Generating universal news summary for article text...")

    # Role priming and audience conditioning produce neutral analysis that works across domains.
    # Exact headers protect the Streamlit card parser from provider formatting drift.
    # Decomposed instructions keep summary, insights, and sentiment distinct without chain-of-thought.

    prompt = (
    "You are a senior cross-domain news analyst writing for a broad professional audience. "
    "Your job is to extract the article's core meaning, practical relevance, major stakeholders, and measured sentiment.\n\n"

    "Output rules:\n"
 
"- Use the exact markdown section headers below, in the exact order.\n"
"- Do not add, remove, rename, or reorder sections.\n"
"- Keep the tone neutral, analytical, and readable for non-specialists.\n"
"- Be concise but specific; avoid generic wording, hype, repetitive phrasing, or unnecessary abstraction.\n"
"- Preserve the exact names of key people, companies, governments, researchers, organizations, and locations mentioned in the article.\n"
"- Always mention the primary people or organizations involved using their proper names in the Summary section.\n"
"- Do not replace named individuals or organizations with descriptive aliases such as 'the pop star', 'the company', 'the singer', 'executives', 'officials', or 'sources'.\n"
"- Mention important stakeholders, decision-makers, spokespersons, or affected groups whenever relevant.\n"
"- Retain important factual details that explain the significance, consequences, or broader context of the event.\n"
"- Base sentiment on the article's tone and likely real-world impact, not on dramatic interpretation or exaggeration.\n\n"



    "Required output structure:\n\n"

    "### 1. Summary\n"
    "Write 2-3 clear sentences explaining:\n"
    "- what happened,\n"
    "- who was involved,\n"
    "- why it matters,\n"
    "- and what broader implications it may have.\n\n"

    "### 2. Key Insights\n"
    "Provide 3 concise bullets covering:\n"
    "- major announcements or actions,\n"
    "- important stakeholders or affected groups,\n"
    "- and broader industry, political, economic, or societal relevance.\n\n"

    "### 3. Sentiment\n"
    "Classify as Positive, Neutral, or Negative, followed by one short sentence explaining the classification.\n\n"

    "Article Content:\n"
    f"{truncated_text}\n")


    try:
        result = generate_response(prompt)

        if result["success"]:
            logger.info(f"Successfully generated universal summary using provider: {result['provider']}")
            summary_text = result["response"]
            if summary_text:
                summary_text = summary_text.replace("\\n", "\n")
            return {
                "success": True,
                "provider": result["provider"],
                "summary": summary_text,
                "error": None
            }

        error_msg = f"LLM Routing failed to generate response: {result['error']}"
        logger.error(error_msg)
        return {
            "success": False,
            "provider": None,
            "summary": None,
            "error": error_msg
        }

    except Exception as e:
        error_msg = f"Unexpected error in universal_summarizer: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "provider": None,
            "summary": None,
            "error": error_msg
        }

# if __name__ == "__main__":
#     import json
    
#     print("Testing Summarizer Module...")
    
#     # Sample article text for local validation (GPT-4o mini announcement summary)
#     sample_text = (
#         "OpenAI has officially released GPT-4o mini, a smaller, highly cost-efficient version of its flagship "
#         "GPT-4o model. This new model is priced at 15 cents per million input tokens and 60 cents per million "
#         "output tokens, which is over 60% cheaper than the previous GPT-3.5 Turbo model. Despite its small size, "
#         "GPT-4o mini scores 82% on the MMLU benchmark, surpassing many other small models in the industry. "
#         "It supports both text and vision inputs, with support for video and audio inputs planned for the near future. "
#         "Developers are expected to use GPT-4o mini for high-volume tasks, multi-step agent chains, and real-time "
#         "applications that require low latency and high affordability. Industry experts suggest this release will "
#         "accelerate the development of agentic workflows and make AI accessibility much more practical for startups."
#     )
    
#     print("\nRunning summary on sample text:")
#     res = summarize_article(sample_text)
#     if res["success"]:
#         print(f"\nProvider Used: {res['provider']}")
#         print("\nGenerated Summary:\n")
#         print(res["summary"])
#     else:
#         print(f"\nFailed to generate summary. Error: {res['error']}")
