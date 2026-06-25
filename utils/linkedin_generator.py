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
    
    # Construct the summary prompt with strict structure, tone, and analysis guidelines
    prompt = (
        "You are an expert AI industry analyst. Analyze the following news article and provide a professional, "
        "concise summary structured exactly as outlined below. Use clear formatting, avoid excessive verbosity, "
        "and optimize your analysis for industry insights.\n\n"
        "Please structure your output exactly as follows:\n\n"
        "### 1. Concise Summary\n"
        "[Provide a clear, brief 2-3 sentence overview of the article's main topic and news]\n\n"
        "### 2. Key Insights\n"
        "[Provide 3 high-impact bullet points outlining the most critical facts, data points, or announcements]\n\n"
        "### 3. Industry Impact\n"
        "[Provide a short paragraph analyzing how this news affects the AI industry, market dynamics, developers, or companies]\n\n"
        "### 4. Future Implications\n"
        "[Provide 1-2 sentences on what this means for the future, next steps, or long-term trends]\n\n"
        "-----------------------------------------\n"
        f"Article Content:\n{truncated_text}\n"
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

    prompt = (
        "You are an expert news analyst. Analyze the following article from any topic area and provide a "
        "professional, concise summary structured exactly as outlined below. Use clear formatting, avoid "
        "excessive verbosity, and optimize your analysis for accurate, broadly useful insights.\n\n"
        "Please structure your output exactly as follows:\n\n"
        "### 1. Concise Summary\n"
        "[Provide a clear, brief 2-3 sentence overview of the article's main topic and news]\n\n"
        "### 2. Key Insights\n"
        "[Provide 3 high-impact bullet points outlining the most critical facts, data points, or announcements]\n\n"
        "### 3. Industry Impact\n"
        "[Provide a short paragraph analyzing how this news affects the relevant industry, community, market, policy area, or stakeholders]\n\n"
        "### 4. Future Implications\n"
        "[Provide 1-2 sentences on what this means for the future, next steps, or long-term trends]\n\n"
        "-----------------------------------------\n"
        f"Article Content:\n{truncated_text}\n"
    )

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

if __name__ == "__main__":
    import json
    
    print("Testing Summarizer Module...")
    
    # Sample article text for local validation (GPT-4o mini announcement summary)
    sample_text = (
        "OpenAI has officially released GPT-4o mini, a smaller, highly cost-efficient version of its flagship "
        "GPT-4o model. This new model is priced at 15 cents per million input tokens and 60 cents per million "
        "output tokens, which is over 60% cheaper than the previous GPT-3.5 Turbo model. Despite its small size, "
        "GPT-4o mini scores 82% on the MMLU benchmark, surpassing many other small models in the industry. "
        "It supports both text and vision inputs, with support for video and audio inputs planned for the near future. "
        "Developers are expected to use GPT-4o mini for high-volume tasks, multi-step agent chains, and real-time "
        "applications that require low latency and high affordability. Industry experts suggest this release will "
        "accelerate the development of agentic workflows and make AI accessibility much more practical for startups."
    )
    
    print("\nRunning summary on sample text:")
    res = summarize_article(sample_text)
    if res["success"]:
        print(f"\nProvider Used: {res['provider']}")
        print("\nGenerated Summary:\n")
        print(res["summary"])
    else:
        print(f"\nFailed to generate summary. Error: {res['error']}")
