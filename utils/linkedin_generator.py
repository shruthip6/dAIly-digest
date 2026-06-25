import logging
import sys
from typing import Dict, Any

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeErrors when printing emojis in Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("linkedin_generator")

# Import the LLM routing orchestrator from llm_router.py
# Bound as generate_response to satisfy the import architectural requirement
try:
    from utils.llm_router import route_prompt as generate_response
except ImportError:
    from llm_router import route_prompt as generate_response

def generate_linkedin_post(summary_text: str) -> Dict[str, Any]:
    """
    Generates a professional LinkedIn post from an AI news summary using routed LLM providers.
    
    Args:
        summary_text (str): The structured summary text of the article.
        
    Returns:
        Dict[str, Any]: A standardized response dictionary:
            {
                "success": bool,
                "provider": str or None,
                "linkedin_post": str or None,
                "error": str or None
            }
    """
    # Gracefully handle empty or invalid summary text inputs
    if not summary_text or not isinstance(summary_text, str) or not summary_text.strip():
        logger.error("Invalid input: Summary text is empty or not a string.")
        return {
            "success": False,
            "provider": None,
            "linkedin_post": None,
            "error": "Summary text is empty or not a valid string."
        }
        
    logger.info("Generating LinkedIn post from summary...")
    
    # Role priming and audience conditioning steer the model toward AI industry thought leadership.
    # Structured writing constraints improve consistency while preserving a natural LinkedIn voice.
    # Quality controls reduce hype, fluff, and provider-specific formatting drift.
    prompt = (
        "You are an AI industry analyst and technical content strategist writing for AI engineers, "
        "founders, researchers, and tech leaders on LinkedIn.\n\n"
        "Transform the structured AI news summary into a concise thought-leadership post that explains "
        "the strategic signal behind the news.\n\n"
        "Writing requirements:\n"
        "- Open with a strong, specific hook about the AI industry shift or technical development.\n"
        "- Explain the core update in plain professional language.\n"
        "- Highlight the most important technical, business, research, or infrastructure implication.\n"
        "- Add a future-oriented perspective grounded in the summary.\n"
        "- Keep paragraphs short and readable on LinkedIn.\n"
        "- Use bullets only if they improve clarity; avoid bullet overload.\n"
        "- Avoid hype language, motivational fluff, clickbait, generic buzzwords, and exaggerated claims.\n"
        "- Avoid excessive emojis; use at most one only if it adds clarity.\n"
        "- Do not use markdown headings, section labels, or XML-style tags.\n"
        "- Keep the post polished, human, and concise.\n\n"
        "Article Summary:\n"
        f"{summary_text}\n"
    )
    
    try:
        # Route prompt to the active/fallback LLM providers
        result = generate_response(prompt)
        
        if result["success"]:
            logger.info(f"Successfully generated LinkedIn post using provider: {result['provider']}")
            post_text = result["response"]
            if post_text:
                # Post-process to convert any literal "\\n" string sequences into actual newlines
                post_text = post_text.replace("\\n", "\n")
            return {
                "success": True,
                "provider": result["provider"],
                "linkedin_post": post_text,
                "error": None
            }
        else:
            error_msg = f"LLM Routing failed to generate response: {result['error']}"
            logger.error(error_msg)
            return {
                "success": False,
                "provider": None,
                "linkedin_post": None,
                "error": error_msg
            }
            
    except Exception as e:
        error_msg = f"Unexpected error in linkedin_generator module: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "provider": None,
            "linkedin_post": None,
            "error": error_msg
        }

def generate_universal_linkedin_post(summary_text: str) -> Dict[str, Any]:
    """
    Generates a professional LinkedIn post from any news summary using routed LLM providers.
    
    Args:
        summary_text (str): The structured summary text of the article.
        
    Returns:
        Dict[str, Any]: A standardized response dictionary:
            {
                "success": bool,
                "provider": str or None,
                "linkedin_post": str or None,
                "error": str or None
            }
    """
    # Gracefully handle empty or invalid summary text inputs
    if not summary_text or not isinstance(summary_text, str) or not summary_text.strip():
        logger.error("Invalid input: Summary text is empty or not a string.")
        return {
            "success": False,
            "provider": None,
            "linkedin_post": None,
            "error": "Summary text is empty or not a valid string."
        }
        
    logger.info("Generating LinkedIn post from summary...")
    
    # Role priming establishes professional news analysis rather than generic social copywriting.
    # Task decomposition gives the post a reliable hook-context-relevance-implication flow.
    # Cross-provider constraints avoid fragile formatting and keep the result LinkedIn-ready.
    prompt = (
        "You are a professional LinkedIn content strategist and cross-domain industry analyst. "
        "Write for professionals, students, tech learners, founders, and a general LinkedIn audience.\n\n"
        "Transform the news summary into a polished LinkedIn post with this narrative flow:\n"
        "1. A strong opening hook that names the concrete development or tension.\n"
        "2. A concise explanation of what happened.\n"
        "3. The broader relevance for professionals, markets, policy, technology, or society.\n"
        "4. A grounded closing insight or question that invites reflection.\n\n"
        "Quality requirements:\n"
        "- Sound natural, intelligent, and human-written.\n"
        "- Keep the post concise, educational, and easy to scan.\n"
        "- Prefer specific details from the summary over generic commentary.\n"
        "- Avoid clickbait, hype, motivational fluff, and excessive corporate buzzwords.\n"
        "- Avoid excessive emojis; use at most one only if genuinely useful.\n"
        "- Do not use markdown headings, section labels, or XML-style tags.\n"
        "- Avoid hashtag clutter; include no hashtags unless one is clearly relevant.\n"
        "- Use clean paragraph spacing and no bullet overload.\n\n"
        "Article Summary:\n"
        f"{summary_text}\n"
    )
    
    try:
        # Route prompt to the active/fallback LLM providers
        result = generate_response(prompt)
        
        if result["success"]:
            logger.info(f"Successfully generated LinkedIn post using provider: {result['provider']}")
            post_text = result["response"]
            if post_text:
                # Post-process to convert any literal "\\n" string sequences into actual newlines
                post_text = post_text.replace("\\n", "\n")
            return {
                "success": True,
                "provider": result["provider"],
                "linkedin_post": post_text,
                "error": None
            }
        else:
            error_msg = f"LLM Routing failed to generate response: {result['error']}"
            logger.error(error_msg)
            return {
                "success": False,
                "provider": None,
                "linkedin_post": None,
                "error": error_msg
            }
            
    except Exception as e:
        error_msg = f"Unexpected error in linkedin_generator module: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "provider": None,
            "linkedin_post": None,
            "error": error_msg
        }

# if __name__ == "__main__":
#     print("Testing LinkedIn Generator Module...")
    
#     # Sample input summary (matching the structure from summarizer.py)
#     sample_summary = (
#         "### 1. Concise Summary\n"
#         "OpenAI has released GPT-4o mini, a smaller and highly cost-efficient AI model. "
#         "It is priced significantly lower than previous models like GPT-3.5 Turbo while achieving an "
#         "impressive score of 82% on the MMLU benchmark, beating other small models in the market.\n\n"
#         "### 2. Key Insights\n"
#         "- Input tokens are priced at $0.15 per million, and output tokens at $0.60 per million (60%+ cheaper than GPT-3.5 Turbo).\n"
#         "- Retains high benchmark capability (82% MMLU) despite the reduced model size and costs.\n"
#         "- Supports text and vision modalities with low-latency responsiveness.\n\n"
#         "### 3. Industry Impact\n"
#         "This release lowers the financial barrier for AI application development, enabling startups and "
#         "enterprises to run high-volume workflows, complex agent chains, and real-time assistants at a fraction of the cost.\n\n"
#         "### 4. Future Implications\n"
#         "As frontier intelligence becomes commoditized and cheap, the focus will rapidly pivot from model building "
#         "to building reliable, complex agent architectures and customer-facing integration layers."
#     )
    
#     print("\nRunning post generation on sample summary:")
#     res = generate_linkedin_post(sample_summary)
#     if res["success"]:
#         print(f"\nProvider Used: {res['provider']}")
#         print("\nGenerated LinkedIn Post:\n")
#         print(res["linkedin_post"])
#     else:
#         print(f"\nFailed to generate post. Error: {res['error']}")